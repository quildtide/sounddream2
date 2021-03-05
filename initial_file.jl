using Flux
using CUDA
using Statistics

using BSON: @save, @load

import WAV
import CSV
import Tables
import Zygote

const GENRE_TYPE = Int8
const BASE_BATCH_SIZE = 2646000
const CHANNEL_COUNT = 2
const DATA_FOL = "data/aesthetics"
const GENRE_COUNT = 3

function get_metadata()
    data = CSV.File("data/aesthetics/processed/info.csv")

    cols = data |> Tables.columntable

    lengths = unique(cols.length)
    genres = [0, 1, 2]

    select_dict = Dict{Int64, Dict{GENRE_TYPE, Vector{String}}}()

    for len in lengths
        push!(select_dict, len => Dict{GENRE_TYPE, Vector{String}}())
        for genre in genres
            push!(select_dict[len], genre => Vector{String}())
        end
    end

    for row in data
        push!(select_dict[row.length][row.genres], row.filename)
    end

    true_genres = Dict{String, GENRE_TYPE}(row.filename => row.genres for row in data)

    return (select_dict, true_genres, lengths, genres)
end

batch_size(len) = trunc(Int64, BASE_BATCH_SIZE / len)


function dataloader(select_dict, chan, lengths, genres)
    len = rand(lengths)

    bsz = batch_size(len)

    x = zeros(Float32, (len + 1, 1, CHANNEL_COUNT, bsz))
    y = zeros(Float32, (length(genres), bsz))

    for i in 1:bsz
        genre = rand(genres)
        filename = rand(select_dict[len][genre])
        y[genre + 1, i] = 1

        audio = Float32.(WAV.wavread(joinpath(DATA_FOL, filename))[1])

        for c in 1:CHANNEL_COUNT
            x[1:size(audio)[1], :, c, i] = audio[:,c]
        end
    end

    put!(chan, (x, y))
    return nothing
end

MODEL = Chain(
            Conv((15,1), CHANNEL_COUNT => 16, identity, dilation = (9, 1), pad = SamePad()),
            Dropout(.25),
            MeanPool((32, 1)),
            swish,
            Conv((15,1), 16 => 64, identity, dilation = (9, 1), pad = SamePad()),
            Dropout(.35),
            MeanPool((8, 1)),
            swish,
            Conv((9,1), 64 => 128, identity, dilation = (7, 1), pad = SamePad()),
            Dropout(.45),
            MeanPool((4, 1)),
            swish,
            Conv((1,1), 128 => GENRE_COUNT, identity, dilation = (5, 1)),
            Dropout(.5),
            AdaptiveMeanPool((1, 1)), # guaranteed dims: (256, 1, 128, B)
            flatten,
            logsoftmax
        ) |> gpu


OPTIMIZER = Flux.Optimise.Optimiser(ClipValue(1e-3), Momentum())

# OPTIMIZER = Flux.Optimise.Momentum()

function sounddreamer_train(reps = 1000, m_iv = 100)
    Flux.trainmode!(MODEL)

    select_dict, true_genres, lengths, genres = get_metadata()

    function dataloader_helper(chan::Channel)
        for i in 1:reps
            dataloader(select_dict, chan, lengths, genres)
        end
    end

    datachannel = Channel(dataloader_helper)

    loss_hist = zeros(m_iv)

    for i in 1:reps
        x, y = take!(datachannel)

        x = x |> gpu
        y = y |> gpu

        ps = params(MODEL)

        loss, back = @sync Zygote.pullback(() -> Flux.Losses.logitcrossentropy(MODEL(x), y), ps)

        gs = back(one(loss))
        Flux.Optimise.update!(OPTIMIZER, ps, gs)

        loss_hist[i%m_iv + 1] = loss

        if i%m_iv == 0
            println("Loop $i--last $m_iv average training loss: $(mean(loss_hist))")
        end
    end

end

function sounddreamer_continuous()
    while time() < 1608135358
        sounddreamer_train(5000, 5000)
        weights = deepcopy(params(cpu(MODEL)))
        @save "auto_store_params/checkpoint_$(time()).bson" weights
    end
end

function modify1_obj(y_hat)
    mean(y_hat)
end

modify2_obj(y_hat) = sum(y_hat .* cu([1,0,0,0]))

MODIFIZER = Flux.Optimise.Descent()

function normalize(x, scale)
    curr_scale = maximum(abs.(x))
    return x / curr_scale * scale
end


function sounddreamer_modify(filename, outname, reps = 100, layer = 17, m_iv = 10, mod_obj = modify2_obj)
    Flux.testmode!(MODEL)

    orig = Float32.(WAV.wavread(joinpath("to_be_modified", filename))[1])

    orig_down = orig[1:4:end,:]

    len = size(orig_down)[1]

    x = zeros(Float32, (len, 1, CHANNEL_COUNT, 1)) |> gpu

    for c in 1:CHANNEL_COUNT 
        x[:, :, c, :] = orig_down[:,c] 
    end

    loss_hist = zeros(m_iv)

    for i in 1:reps
        l4_μ = deepcopy(MODEL[4].μ)
        l8_μ = deepcopy(MODEL[8].μ) 
        l12_μ = deepcopy(MODEL[12].μ)
        l4_σ² = deepcopy(MODEL[4].σ²)
        l8_σ² = deepcopy(MODEL[8].σ²) 
        l12_σ² = deepcopy(MODEL[12].σ²)

        ps = Flux.params(x)

        loss, back = @sync Zygote.pullback(() -> mod_obj(MODEL[1:layer](x)), ps)

        gs = back(one(loss))

        x = x .+ gs[x]

        loss_hist[i%m_iv + 1] = loss

        if i%m_iv == 0
            println("Loop $i--last $m_iv average modification loss: $(mean(loss_hist))")
        end
        
        MODEL[4].μ = l4_μ
        MODEL[8].μ = l8_μ
        MODEL[12].μ = l12_μ
        MODEL[4].σ² = l4_σ²
        MODEL[8].σ² = l8_σ²
        MODEL[12].σ² = l12_σ²
    end

    base_filename = "modif_out/$(outname)_l$(layer)_r$(reps)"

    x_out = reshape(cpu(x), (len, 2))

    WAV.wavwrite(x_out .- orig_down, base_filename * "_residual.wav", Fs=11025)

    WAV.wavwrite(x_out, base_filename * ".wav", Fs=11025)
end

function check_exp(filename)
    Flux.testmode!(MODEL)

    last_mu = MODEL[4].μ |> cpu
    last_std = MODEL[4].σ² |> cpu
    last_bias = MODEL[4].β |> cpu
    last_scale = MODEL[4].γ |> cpu

    orig = Float32.(WAV.wavread(joinpath("to_be_modified", filename))[1])

    orig_down = orig[1:4:end,:]

    len = size(orig_down)[1]

    x = zeros(Float32, (len, 1, CHANNEL_COUNT, 1)) |> gpu

    for c in 1:CHANNEL_COUNT 
        x[:, :, c, :] = orig_down[:,c] 
    end

    results = exp.(cpu(MODEL(x)))

    curr_mu = MODEL[4].μ |> cpu
    curr_std = MODEL[4].σ² |> cpu
    curr_bias = MODEL[4].β |> cpu
    curr_scale = MODEL[4].γ |> cpu

    if last_mu != curr_mu println("mu has changed") end
    if last_std != curr_std println("std has changed") end
    if last_bias != curr_bias println("bias has changed") end
    if last_scale != curr_scale println("scale has changed") end

    println("Expected Genre (FG, VW, CT, YT): $results")
end

function sounddreamer_test(reps = 1000)
    Flux.testmode!(MODEL)

    select_dict, true_genres, lengths, genres = get_metadata()

    function dataloader_helper(chan::Channel)
        for i in 1:reps
            dataloader(select_dict, chan, lengths, genres)
        end
    end

    datachannel = Channel(dataloader_helper)

    conf_matrix = zeros(Int, (4, 4, 5))

    length_dict = Dict(sort(lengths)[i] => i for i in 1:length(lengths)) 

    for j in 1:reps
        x, y = take!(datachannel)

        x = x |> gpu
        
        y_hat = cpu(MODEL(x))

        # show(stdout, "text/plain", y)
        # print("\n")
        # show(stdout, "text/plain", y_hat)
        # print("\n")

        bsz = size(x)[4]

        true_y = [findmax(y, dims = 1)[2][i][1] for i in 1:bsz]
        exp_y = [findmax(y_hat, dims = 1)[2][i][1] for i in 1:bsz]
        len = length_dict[size(x)[1] - 1]

        # println(true_y)
        # println(exp_y)
        # println(len)

        for i in 1:bsz
            conf_matrix[true_y[i], exp_y[i], len] += 1
        end
    end

    show(stdout, "text/plain", conf_matrix)
end
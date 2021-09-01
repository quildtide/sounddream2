using Flux

include("constants.jl")

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
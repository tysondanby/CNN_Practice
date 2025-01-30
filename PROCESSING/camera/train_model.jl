using Metalhead, Flux
previously_loaded = false

if previously_loaded == false
    model = ResNet(18; pretrain = true)
else
end
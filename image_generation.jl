using Images
#TODO: put this in the math.jl file
function magnitude(x::Tuple{T,T}) where T <: Any
    return sqrt(x[1]^2 + x[2]^2)
end
function drawline!(img,thickness,start,finish;shade = Gray(0.0))
    for i = 1:1:size(img)[1]
       for j = 1:1:size(img)[2]
        pt = (i,j)
        vec1 = pt .- start
        vec2 = finish .- start
        dist = magnitude(vec1)
        #println(sum(vec1 .* vec2)/(magnitude(vec1)*magnitude(vec2)))
        angle = acos(sum(vec1 .* vec2)/(magnitude(vec1)*magnitude(vec2)))
        distancealongline = dist*cos(angle)/magnitude(vec2)
        distancefromline = dist*sin(angle)
        if ((distancealongline >= 0)&&(distancealongline <= 1)&& (abs(distancefromline) <= 0.5*thickness))
            img[i,j] = shade
        end
       end
    end
end

function randomlinepicture(;dims = (100, 100))
    lineshade = rand(Float32)
    backgroundshade = rand(Float32)
    while lineshade == backgroundshade
        backgroundshade = rand(Float32)
    end
    thickness = rand()*6

    img = Gray.(Float32.(ones(100,100))*backgroundshade)
    x1 = Float32(round(rand()*dims[1]))
    x2 = Float32(round(rand()*dims[1]))
    y1 = Float32(round(rand()*dims[2]))
    y2 = Float32(round(rand()*dims[2]))
    drawline!(img,thickness,(x1,y1),(x2,y2); shade = lineshade)
    return img, ((x1,y1),(x2,y2))
end

function randomlinepictures(n;dims = (100, 100))
    testimg = Array{Gray{Float32},2}(undef,dims[1],dims[2])
    imgs = Vector{typeof(testimg)}(undef,n)
    datas = Vector{Tuple{Tuple{Float32,Float32},Tuple{Float32,Float32}}}(undef,n)
    for i = 1:1:n
        imgs[i],datas[i] = randomlinepicture(dims = size(testimg))
    end
    return imgs, datas
end

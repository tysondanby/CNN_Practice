#-----ENVIRONMENT SETUP-------------------------------------------------------
using Pkg
Pkg.activate(@__DIR__) 
required_packages = ["GLMakie", "Images", "CSV", "DataFrames"]
function is_installed(pkg)
    return haskey(Pkg.project().dependencies, pkg)
end
function ensure_packages(packages)
    for pkg in packages
        if !is_installed(pkg)
            println("Installing missing package: ", pkg)
            Pkg.add(pkg)
        end
    end
end
ensure_packages(required_packages)
Pkg.instantiate()
working_dir = dirname(dirname(dirname(@__DIR__)))
cd(working_dir)
#-----END ENVIRONMENT SETUP---------------------------------------------------

using GLMakie, Images, CSV, DataFrames
include(pwd()*"/LIB/arrayoperations.jl")
include(pwd()*"/LIB/math.jl")
threebladed = [4 8]
imagedirectory = "DATA/camera/prepared"

function getpropellerfromimagename(imagename)
    firstunderscore = findfirst(x -> x == '_', imagename)
    secondunderscore = findfirst(x -> x == '_', imagename[firstunderscore+1:end]) + firstunderscore
    return parse(Int32,imagename[firstunderscore+1:secondunderscore-1])
end

function savemanualdata(imagename,points)
    traincsv = CSV.File("DATA/camera/model_data/train.csv")
    verifycsv = CSV.File("DATA/camera/model_data/verify.csv")
    if length(verifycsv) < length(traincsv)
        push!(verifycsv.filename,imagename)
        push!(verifycsv.x1,points[1][1])
        push!(verifycsv.y1,points[1][2])
        push!(verifycsv.x2,points[2][1])
        push!(verifycsv.y2,points[2][2])
        push!(verifycsv.x3,points[3][1])
        push!(verifycsv.y3,points[3][2])
        CSV.write("DATA/camera/model_data/verify.csv",DataFrame(verifycsv))
    else
        push!(traincsv.filename,imagename)
        push!(traincsv.x1,points[1][1])
        push!(traincsv.y1,points[1][2])
        push!(traincsv.x2,points[2][1])
        push!(traincsv.y2,points[2][2])
        push!(traincsv.x3,points[3][1])
        push!(traincsv.y3,points[3][2])
        CSV.write("DATA/camera/model_data/train.csv",DataFrame(traincsv))
    end
end

function circumscribepropeller!(scene,points,propeller)
    center = pointscenter(points)
    radius = sqrt(sum((points[1] .- center).^2))
    if !contains(threebladed, propeller)
        center = pointscenter(points[2:3])
        radius = sqrt(sum((points[2] .- center).^2))
    end
    thetas = collect(0:2*pi/20:2*pi)
    circlepoints = [center]
    for i = 1:1:length(thetas)
        push!(circlepoints,(cos(thetas[i])*radius + center[1], sin(thetas[i])*radius + center[2]))
    end
    scatter!(scene, circlepoints, color = :blue)
end

function reorderclockwisefromtop(points)
    newpoints = similar(points)
    if (points[1][1] <= 1.0) && (points[1][2] <= 1.0)
        newpoints[1] = points[1]
        if points[2][2] > points[3][2]#higher y
            newpoints[2:3] = points[2:3]
        else
            newpoints[2] = points[3]
            newpoints[3] = points[2]
        end
    else
        ys = [points[1][2], points[2][2], points[3][2]]
        xs = [points[1][1], points[2][1], points[3][1]]
        topindex = findfirst(x->x == maximum(ys),ys)
        xs[topindex] = 0.0
        nextindex = findfirst(x->x == maximum(xs),xs)
        lastindex = findfirst(x->((x != maximum(xs))&&(x != minimum(xs))),xs)
        newpoints[1] = points[topindex]
        newpoints[2] = points[nextindex]
        newpoints[3] = points[lastindex]
    end
    return newpoints
end

function characterizesingleimage()
    traincsv = CSV.File("DATA/camera/model_data/train.csv")
    verifycsv = CSV.File("DATA/camera/model_data/verify.csv")
    imagename = selectrandom(readdir(imagedirectory))
    println(imagename)
    while (imagename in traincsv) || (imagename in traincsv)
        imagename = selectrandom(readdir(imagedirectory))
    end
    img = load(imagedirectory*"/"*imagename)
    propeller = getpropellerfromimagename(imagename)

    #WINDOW DIALOG
    scene = Scene(camera=campixel!,show_axis=false, size = size(img))#, padding = (10.0, 10.0))
    image!(scene, img, interpolate=false)#, padding = (0.0, 0.0))
    screen = display(scene)
    points = Vector{Tuple{Float64, Float64}}([])
    npoints = 0
    if !contains(threebladed, propeller)
        push!(points, (0.0,0.0))
        npoints = 1
    end
    npointsreset = deepcopy(npoints)
    pointsreset = deepcopy(points)
    on(scene.events.mousebutton) do button
        if ispressed(scene,Mouse.left)
            if (scene.events.mouseposition[][1] >= 1) && (scene.events.mouseposition[][2] >= 1)
                if npoints < 3
                    mp = scene.events.mouseposition[]
                    push!(points, mp)
                    npoints = npoints + 1
                    scatter!(scene, points, color = :red, marker = '+')
                    if npoints == 3
                        circumscribepropeller!(scene,points,propeller) 
                    end
                else
                    @async begin
                        sleep(0.1)
                        GLMakie.close(screen)
                    end
                end
            end
        end
        if ispressed(scene,Mouse.right)
            npoints = npointsreset
            points = deepcopy(pointsreset)
            image!(scene, img, interpolate=false)#, padding = (0.0, 0.0))
        end
    end
    println("Click on the propeller tips. Right click to cancel, left click to confirm.")
    #END WINDOW DIALOG

    wait(screen)
    if length(points) == 3
        fixedpoints = reorderclockwisefromtop(points)
        if fixedpoints != points
            println("Points rearranged")
            println(fixedpoints)
        end
        savemanualdata(imagename,fixedpoints)
        println("Datapoint recorded")
    else
        println("Datapoint aborted; insufficient points")
    end 
end

function characterizeimages(n)
    for i = 1:1:n
        characterizesingleimage()
        println("$i / $n")
        println()
        println()
    end
end
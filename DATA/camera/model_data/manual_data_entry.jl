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
include(pwd()*"/LIB/imageprocessing.jl")
threebladed = [4 8]
imagedirectory = "DATA/camera/prepared"

function getpropellerfromimagename(imagename)
    firstunderscore = findfirst(x -> x == '_', imagename)
    secondunderscore = findfirst(x -> x == '_', imagename[firstunderscore+1:end]) + firstunderscore
    return parse(Int32,imagename[firstunderscore+1:secondunderscore-1])
end

function getpropellerspeedimagenumberfromimagename(imagename)
    firstunderscore = findfirst(x -> x == '_', imagename)
    secondunderscore = findfirst(x -> x == '_', imagename[firstunderscore+1:end]) + firstunderscore
    thirdunderscore = findfirst(x -> x == '_', imagename[secondunderscore+1:end]) + firstunderscore + secondunderscore
    return parse(Int32,imagename[firstunderscore+1:secondunderscore-1]),parse(Int32,imagename[secondunderscore+1:thirdunderscore-1]),parse(Int32,imagename[thirdunderscore+1:end-4])
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
    while (imagename in traincsv) || (imagename in verifycsv)
        imagename = selectrandom(readdir(imagedirectory))
        println("Duplicate, selecting new.")
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

function resultsfromcsv(imagename)
    resultscsv = CSV.File("DATA/camera/model_data/model_results.csv")
    i = findfirst(x -> x == imagename, resultscsv.filename)
    return resultscsv.xc[i],resultscsv.yc[i],resultscsv.r[i],resultscsv.sin[i],resultscsv.cos[i]
end

function plotresults(imagename)
    xc,yc,r,sinth,costh = resultsfromcsv(imagename)
    img = load(imagedirectory*"/"*imagename)
    propeller = getpropellerfromimagename(imagename)
    if (propeller == 1) || (propeller == 5)
        flipimageX!(img)
    end
    #WINDOW DIALOG
    scene = Scene(camera=campixel!,show_axis=false, size = size(img))#, padding = (10.0, 10.0))
    image!(scene, img, interpolate=false)#, padding = (0.0, 0.0))
    thetas = collect(0.0:5.0:360.0)
    circlepoints = [(xc,yc)]
    for i = 1:1:length(thetas)
        push!(circlepoints, ( r*sind(thetas[i])+xc , r*cosd(thetas[i])+yc ))
    end
    ntheta = atand(sinth,costh)
    gamma = 0.0
    if contains(threebladed, propeller)
        gamma = (ntheta/3) + 30
    else
        gamma = ntheta/2
    end
    rs = collect(0.1:0.05:1.0)
    linepoints = [(xc,yc)]
    for i = 1:1:length(rs)
        push!(linepoints, ((1-rs[i]) .* ( xc , yc )) .+ (rs[i] .* (xc + r*sind(gamma), yc + r*cosd(gamma))))
    end
    scatter!(scene, circlepoints, color = :blue, marker = '.')
    scatter!(scene, linepoints, color = :red, marker = '.')
    screen = display(scene)
    on(scene.events.mousebutton) do button
        if ispressed(scene,Mouse.left)
            if (scene.events.mouseposition[][1] >= 1) && (scene.events.mouseposition[][2] >= 1)
                println("Image good")
                @async begin
                    sleep(0.1)
                    GLMakie.close(screen)
                end
            end
        end
        if ispressed(scene,Mouse.right)
            println("Image not good, manually correct it")
            @async begin
                sleep(0.1)
                GLMakie.close(screen)
            end
        end
    end
    wait(screen)
end

function visualizeresults(imagenumbers)
    traincsv = CSV.File("DATA/camera/model_data/train.csv")
    resultscsv = CSV.File("DATA/camera/model_data/model_results.csv")
    trainedimagenames = traincsv.filename[2:end]
    resultsimagenames = resultscsv.filename[2:end]
    for i = 1:1:length(imagenumbers)
        imagename = resultsimagenames[imagenumbers[i]]
        if imagename in trainedimagenames
            println("Trained Image")
        end
        plotresults(imagename)
    end
end

function getprogress(csvfilename)
    name = (CSV.File("DATA/camera/model_data/manual_results.csv")).filename[end]
    if name[1] == '_'
        return 1, 4500, 1
    else
        return getpropellerspeedimagenumberfromimagename(imagename)
    end
end

function pointsencodestring(points)
    stringout = "B"
    for i = 1:1:length(points)
        xpt,ypt = points[i]
        stringout = stringout*"x$xpt"*"y$ypt"
    end
    return stringout
end

function getpropellertiplocations(imagename,trainfilename,verifyfilename)
    traincsv = CSV.File(trainfilename)
    verifycsv = CSV.File(verifyfilename)
    points = []
    if imagename in traincsv.filename
        i = findfirst(x -> x == imagename, traincsv.filename)
        points = [(verifycsv.x1[i],verifycsv.y1[i]),(verifycsv.x2[i],verifycsv.y2[i]),(verifycsv.x3[i],verifycsv.y3[i])]
        img = load(imagedirectory*"/"*imagename)
        propeller = getpropellerfromimagename(imagename)
        scene = Scene(camera=campixel!,show_axis=false, size = size(img))
        image!(scene, img, interpolate=false)
        circumscribepropeller!(scene,points,propeller)
        scatter!(scene, points, color = :red, marker = '+')
        screen = display(scene)
        on(scene.events.mousebutton) do button
            if ispressed(scene,Mouse.left)
                if (scene.events.mouseposition[][1] >= 1) && (scene.events.mouseposition[][2] >= 1)      
                    @async begin
                        sleep(0.1)
                        GLMakie.close(screen)
                    end
                end
            end
        end
        wait(screen)
    elseif imagename in verifycsv.filename
        i = findfirst(x -> x == imagename, verifycsv.filename)
        points = [(verifycsv.x1[i],verifycsv.y1[i]),(verifycsv.x2[i],verifycsv.y2[i]),(verifycsv.x3[i],verifycsv.y3[i])]
        img = load(imagedirectory*"/"*imagename)
        propeller = getpropellerfromimagename(imagename)
        scene = Scene(camera=campixel!,show_axis=false, size = size(img))
        image!(scene, img, interpolate=false)
        circumscribepropeller!(scene,points,propeller)
        scatter!(scene, points, color = :red, marker = '+')
        screen = display(scene)
        on(scene.events.mousebutton) do button
            if ispressed(scene,Mouse.left)
                if (scene.events.mouseposition[][1] >= 1) && (scene.events.mouseposition[][2] >= 1)      
                    @async begin
                        sleep(0.1)
                        GLMakie.close(screen)
                    end
                end
            end
        end
        wait(screen)
    else
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

    if (points[1][1] < 1) && (points[1][2] < 1)
        points = points[2:3]
    end
    return pointscenter(points), points
end

function getcavitationbounds(imagename,center,point)
    sleep(0.3)
    img = load(imagedirectory*"/"*imagename)
    propeller = getpropellerfromimagename(imagename)
    #WINDOW DIALOG
    scene = Scene(camera=campixel!,show_axis=false, size = size(img))
    image!(scene, img, interpolate=false)
    screen = display(scene)
    points = Vector{Tuple{Float64, Float64}}([])
    push!(points,center)
    push!(points,point)
    lines!(scene, points; color = :tomato)
    scatter!(scene, points, color = :blue, marker = '+')
    confirmationflag = false
    on(scene.events.mousebutton) do button
        if ispressed(scene,Mouse.left)
            if (scene.events.mouseposition[][1] >= 1) && (scene.events.mouseposition[][2] >= 1)
                mp = scene.events.mouseposition[]
                if confirmationflag == true
                    @async begin
                        sleep(0.1)
                        GLMakie.close(screen)
                    end
                else
                    push!(points, mp)
                    lines!(scene, points; color = :tomato)
                    scatter!(scene, points, color = :blue, marker = '+')
                end
                if distance(mp[1],mp[2],center[1],center[2]) <= 30
                    confirmationflag = true
                end
            end
        end
        if ispressed(scene,Mouse.right)
            confirmationflag = false
            if length(points) > 2
                points = points[1:end-1]
            end
            image!(scene, img, interpolate=false)
            lines!(scene, points; color = :tomato)
            scatter!(scene, points, color = :blue, marker = '+')
        end
    end
    wait(screen) 
    return points
end

function manualdataentry()
    propellers = [1,2,3,4,5,6,7,8]
    speeds = collect(4500:500:15000)
    imagenumbers = [1,11,21,31,41,51,61,71,81,91]
    currentprop, currentspeed, currentimagenumber = getprogress("DATA/camera/model_data/manual_results.csv")
    resultscsv =CSV.File("DATA/camera/model_data/manual_results.csv")
    proprange = findfirst(x -> x==currentprop,propellers):1:length(propellers)
    speedrange = findfirst(x -> x==currentspeed,speeds):1:length(speeds)
    imagerange = findfirst(x -> x==currentimagenumber,imagenumbers):1:length(imagenumbers)
    for propellerindex = proprange
        propeller = propellers[propellerindex]
        for speedindex = speedrange
            speed = speeds[speedindex]
            for imageindex = imagerange #index 1 to 10
                imagenumber = imagenumbers[imageindex]
                imagename = "DATASET1_$propeller"*"_$speed"*"_"*lpad(imagenumber,5,"0")*".png"
                center, points = getpropellertiplocations(imagename,"DATA/camera/model_data/train.csv","DATA/camera/model_data/verify.csv")
                stringdata = ""
                for (i, point) in enumerate(points)
                    polypoints = getcavitationbounds(imagename,center,point)
                    stringdata = stringdata*pointsencodestring(polypoints)
                end
                push!(resultscsv.filename,imagename)
                push!(resultscsv.data,stringdata)
                CSV.write("DATA/camera/model_data/manual_results.csv",DataFrame(resultscsv))
                println("Saved")
            end
            imagerange = 1:1:length(imagenumbers)
        end
        speedrange = 1:1:length(speeds)
    end
end
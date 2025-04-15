#-----ENVIRONMENT SETUP-------------------------------------------------------
using Pkg
Pkg.activate(@__DIR__) 
required_packages = ["Plots", "CSV", "DataFrames", "FFTW", "WAV", "GaussianProcesses"]
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
println(@__DIR__)
working_dir = dirname(@__DIR__)
cd(working_dir)
#-----END ENVIRONMENT SETUP---------------------------------------------------

using Plots, CSV, DataFrames, FFTW, WAV, Base.Threads, GaussianProcesses
include(pwd()*"/LIB/arrayoperations.jl")
include(pwd()*"/LIB/math.jl")
#include(pwd()*"/LIB/imageprocessing.jl")
threebladed = [4 8]
invert = [1 5]
imagedirectory = "DATA/camera/prepared"
manual_resultscsv = CSV.File("DATA/camera/model_data/manual_results.csv")

macro suppress_output(ex)
    return quote
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                $(esc(ex))
            end
        end
    end
end

function getdatafromimagename(imagename)
    index = findfirst(i->(i==imagename), manual_resultscsv.filename)
    return manual_resultscsv.data[index]
end

function getpointsfromstring(stringofpoints)
    xindex = findfirst(i->(i=='x'), stringofpoints)
    yindex = findfirst(i->(i=='y'), stringofpoints)
    xindicies = []
    yindicies = []
    while xindex != nothing
        push!(xindicies,xindex)
        yindex = findfirst(i->(i=='y'), stringofpoints[(xindex+1):end]) + xindex
        push!(yindicies,yindex)
        step = findfirst(i->(i=='x'), stringofpoints[(xindex+1):end])
        if step != nothing
            xindex = step + xindex
        else
            xindex = nothing
        end
    end
    points = zeros(2,length(xindicies))
    for i = 1:1:(length(xindicies)-1)
        points[1,i] = parse(Float64,stringofpoints[(xindicies[i]+1):(yindicies[i]-1)])
        points[2,i] = parse(Float64,stringofpoints[(yindicies[i]+1):(xindicies[i+1]-1)])
    end
    points[1,end] = parse(Float64,stringofpoints[(xindicies[end]+1):(yindicies[end]-1)])
    points[2,end] = parse(Float64,stringofpoints[(yindicies[end]+1):end])
    return points
end

function makeradial(pointset,propnumber; upsample = 0)
    npoints = size(pointset)[2]
    radialpointset = zeros(2,(npoints-1)*(upsample+1))
    r = distance(pointset[1,1],pointset[2,1],pointset[1,2],pointset[2,2])
    thetabase = atan(pointset[1,2]-pointset[1,1],pointset[2,2]-pointset[2,1])
    for i = 2:1:npoints
        xciel = pointset[1,i]
        yciel = pointset[2,i]
        xprev = pointset[1,i-1]
        yprev = pointset[2,i-1]
        for j = 1:1:(upsample+1)
            frac = j/(upsample+1)
            x = xprev + frac*(xciel-xprev)
            y = yprev + frac*(yciel-yprev)
            index = (i-2)*(upsample+1) + j
            radialpointset[1,index] = distance(pointset[1,1],pointset[2,1],x,y)/r
            radialpointset[2,index] = -1*boundanglepiminuspi(atan(x-pointset[1,1],y-pointset[2,1]) - thetabase)
            
            if contains(invert, propnumber)
                radialpointset[2,index] = -radialpointset[2,index]
            end
            
        end
    end
    return radialpointset
end

function filteroverlapandtip(pointset,rcuttoff)
    npoints = size(pointset)[2]
    newrs = Float64[]
    newths = Float64[]
    tipcavitationflag = false
    rmax = Float64(0.0)
    index = npoints
    endindex = findfirst(x->(x>=0.99),pointset[1,:])
    while (!tipcavitationflag && (index > endindex))#index > 5 worked if you get a bug here
        r = Float64(pointset[1,index])
        th = Float64(pointset[2,index])
        if (npoints-index)>1
            rmax = maximum(newrs)
        else
            rmax = Float64(0.0)
        end
        if (r < rmax) && ((npoints-index)>2)
            hi = findfirst(x->(x>r),newrs)
            lo = hi-1
            newrs = newrs[1:hi]
            newths = newths[1:hi]
            if (r>rcuttoff) && (findfirst(x->(x>th),newths) == nothing) && false 
                tipcavitationflag = true
                push!(newrs,newrs[end])
                push!(newths,2*pi)
                push!(newrs,1.0)
                push!(newths,2*pi)
            else
                newths[end] = (((newths[end]-newths[lo])/(newrs[end]-newrs[lo]))*(r-newrs[lo])) + newths[lo]
                newrs[end] = r
                push!(newrs,r)
                push!(newths,th)
            end
        else
            push!(newrs,r)
            push!(newths,th)
        end
        index = index - 1
    end
    newpointset = zeros(2,length(newrs))
    newpointset[1,:] = newrs
    newpointset[2,:] = newths
    return newpointset
end

function cleanuppointset(pointset;n = 250, rmin = 0.32, rcuttoff = 0.67)
    filteredpointset = filteroverlapandtip(pointset,rcuttoff)
    rclean = collect(range(rmin,1.0; length = n))
    thclean = similar(rclean)
    for (i,r) in enumerate(rclean)
        upperindex = findfirst(x->(x>r), filteredpointset[1,:])
        rhigh = filteredpointset[1,end]
        thhigh = filteredpointset[2,end]
        rlow = 0.0
        thlow = 0.0
        if upperindex != nothing
            rhigh = filteredpointset[1,upperindex]
            thhigh = filteredpointset[2,upperindex]
            if (upperindex - 1) > 0
                rlow = filteredpointset[1,upperindex-1]
                thlow = filteredpointset[2,upperindex-1]
            end
        end
        thclean[i] = (((r-rlow)/(rhigh-rlow)) * (thhigh - thlow)) + thlow
    end
    return rclean,thclean
end

function getpointsfromcsv(imagename,propnumber,npointsout)
    datastring = getdatafromimagename(imagename)
    pointsets = []
    nblades = count(i->(i=='B'), datastring)
    startindex = Int64.(zeros(nblades+1))
    startindex[1] = 1
    startindex[end] = length(datastring) + 1
    for i = 2:1:nblades
        startindex[i] = findfirst(i->(i=='B'), datastring[(startindex[i-1]+1):end]) + startindex[i-1]
    end
    thclean = zeros(nblades,npointsout)
    rclean = zeros(npointsout)
    for i = 1:1:nblades
        pointset = getpointsfromstring(datastring[ (startindex[i]+1):(startindex[i+1]-1) ])
        radialpointset = makeradial(pointset,propnumber; upsample = 5)
        rclean,thclean[i,:] = cleanuppointset(radialpointset; n = npointsout)
        push!(pointsets,radialpointset)
    end
    return rclean, thclean
end

#TODO: verify speed better in this function
function getpropspeeddatafromcsv(propnumber,speed;npoints = 250, percentilethreshold = 0.8)
    aliases = [5000 5500;
               5040 5520]
    speedname = deepcopy(speed)
    if propnumber == 5 #Just correcting a stupid mistake I made with file naming.
        if speed == 5000
            speedname = 5040
        elseif speed == 5500
            speedname = 5520
        end
    end
    imagenumbers = [1,11,21,31,41,51,61,71,81,91] # If I add more samples, this must change.
    rclean = zeros(npoints)
    nblades = 2
    if contains(threebladed, propnumber)
        nblades = 3
    end
    thclean = zeros(nblades*length(imagenumbers),npoints)
    for (i,imagenumber) in enumerate(imagenumbers)
        imagename = "DATASET1_$propnumber"*"_$speedname"*"_"*lpad(imagenumber,5,"0")*".png"
        rclean, thclean[(nblades*(i-1)+1):(nblades*i),:] = getpointsfromcsv(imagename,propnumber,npoints)
    end
    thout = similar(rclean)
    stdev = similar(rclean)
    for (i,r) in enumerate(rclean)
        thout[i], stdev[i] = meanoverpercentile(thclean[:,i],percentilethreshold)
    end
    return rclean, thout .* rclean, stdev .* rclean
end

function plotwitherror(xpts,ypts,stdevpts)
    linecolor = (1.0,0.0,0.0)
    tipvortexcolor = RGBA(0.6,0.7,1.0,1.0)
    yptshigh = ypts + stdevpts
    yptslow = ypts - stdevpts
    shapex = vcat(xpts,reverse(xpts))
    shapey = vcat(yptshigh,reverse(yptslow))
    xbounds = (0,1)
    ybounds = (-0.05,1)
    p = plot(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "")
    plot!(xpts,ypts, label = "Cavitation Bounds", color = RGBA(linecolor... ,1.0))
    plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
    index1 = findfirst(x->(x> 1.0),ypts)
    if index1 == nothing
        rtip = 1.1
    else
        rtip = xpts[index1]
    end
    if rtip < 1.0
        plot!(Shape([rtip,xbounds[2],xbounds[2],rtip],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor, label = "Tip Vortex Cavitation")
    end
    return p
end

function plotwitherrorboth(xpts,ypts,stdevpts,ypts2,stdevpts2)#TODO: edit this to have titles and legend in the correct spot
    linecolor = (1.0,0.0,0.0)
    linecolor2 = (0.0,1.0,0.0)
    tipvortexcolor = RGBA(0.6,0.7,1.0,1.0)
    tipvortexcolor2 = RGBA(0.6,0.7,1.0,1.0)
    yptshigh = ypts + stdevpts
    yptslow = ypts - stdevpts
    shapex = vcat(xpts,reverse(xpts))
    shapey = vcat(yptshigh,reverse(yptslow))
    yptshigh2 = ypts2 + stdevpts2
    yptslow2 = ypts2 - stdevpts2
    shapex2 = vcat(xpts,reverse(xpts))
    shapey2 = vcat(yptshigh2,reverse(yptslow2))
    xbounds = (0,1)
    ybounds = (-0.05,1)
    
    index1 = findfirst(x->(x> 1.0),ypts)
    index2 = findfirst(x->(x> 1.0),ypts2)
    if index1 == nothing
        rtip = 1.1
    else
        rtip = xpts[index1]
    end
    if index2 == nothing
        rtip2 = 1.1
    else
        rtip2 = xpts[index2]
    end
    if rtip < rtip2
        p = plot(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "", legend = :topleft)
        plot!(xpts,ypts, label = "Cavitation Bounds - SH", color = RGBA(linecolor... ,1.0))
        if rtip < 1.0
            # plot!(Shape([rtip,xbounds[2],xbounds[2],rtip],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor, label = "Tip Vortex Cavitation - SH")
        end
        
        plot!(Shape(shapex2,shapey2);color = RGBA(linecolor2... ,0.2),linecolor = RGBA(0,0,0,0), label = "")
        plot!(xpts,ypts2, label = "Cavitation Bounds - Smooth Hydrophillic", color = RGBA(linecolor2... ,1.0))
        if rtip2 < 1.0
            # plot!(Shape([rtip2,xbounds[2],xbounds[2],rtip2],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor2, label = "Tip Vortex Cavitation - Smooth Hydrophillic")
        end
        plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
        return p
    else
        p = plot(Shape(shapex2,shapey2);color = RGBA(linecolor2... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "", legend = :topleft)
        plot!(xpts,ypts2, label = "Cavitation Bounds - Smooth Hydrophillic", color = RGBA(linecolor2... ,1.0))
        if rtip2 < 1.0
            # plot!(Shape([rtip2,xbounds[2],xbounds[2],rtip2],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor2, label = "Tip Vortex Cavitation - Smooth Hydrophillic")
        end

        plot!(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), label = "")
        plot!(xpts,ypts, label = "Cavitation Bounds - SH", color = RGBA(linecolor... ,1.0))
        if rtip < 1.0
           # plot!(Shape([rtip,xbounds[2],xbounds[2],rtip],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor, label = "Tip Vortex Cavitation - SH")
        end
        plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
        return p
    end
end

function plotspeed(propnumber,speed)#Deprecated
    rclean, thout, stdev = getpropspeeddatafromcsv(propnumber,speed)
    rclean, thcontrol, stdev2 = getpropspeeddatafromcsv(propnumber,4500)
    return plotwitherror(rclean, thout .- thcontrol, @. sqrt(stdev^2 + stdev2^2))
end

function plotspeedboth(propnumber,speed)
    rclean, thout, stdev = getpropspeeddatafromcsv(propnumber,speed)
    rclean, thout2, stdev2 = getpropspeeddatafromcsv(propnumber+4,speed)
    return plotwitherrorboth(rclean, thout, stdev,thout2, stdev2)
end

function createvisualfigures()
    speeds = collect(4500:500:15000)
    props = collect(1:1:4)
    for prop in props
        for speed in speeds
            p = plotspeedboth(prop,speed)
            savefig(p,"DATA/output/figures/highspeed_$prop"*"_$speed"*".png")
        end
    end
end

#--------------AUDIO
function pruneslow(ts, speeds, torques)
    newts = []
    newspeeds = []
    newtorques = []
    indicies = []
    for i = 1:1:length(ts)
        if speeds[i] > 4000.0
            push!(newts,ts[i])
            push!(newspeeds,speeds[i])
            push!(newtorques,torques[i])
            push!(indicies,i)
        end
    end
    return newts, newspeeds, newtorques, indicies
end

function prunefastspeedchangeonce(ts, speeds, torques, indicies; delta = 0.0)
    newts = [ts[1]]
    newspeeds = [speeds[1]]
    newtorques = [torques[1]]
    newindicies = [indicies[1]]
    for i = 2:1:(length(ts)-1)
        if (abs(speeds[i] - speeds[i-1]) < delta) #|| (abs(speeds[i] - speeds[i+1]) < 400.0)
            push!(newts,ts[i])
            push!(newspeeds,speeds[i])
            push!(newtorques,torques[i])
            push!(newindicies,indicies[i])
        end
    end
    push!(newts,ts[end])
    push!(newspeeds,speeds[end])
    push!(newtorques,torques[end])
    push!(newindicies,indicies[end])
    return newts, newspeeds, newtorques, newindicies
end

function prunefastspeedchange(ts, speeds, torques,indicies; delta = 0.0)
    oldts = deepcopy(ts)
    oldspeeds = deepcopy(speeds)
    oldtorques = deepcopy(torques)
    newts, newspeeds, newtorques, newindicies = prunefastspeedchangeonce(ts, speeds, torques,indicies; delta = delta)
    while length(newts) != length(oldts)
        oldts = deepcopy(newts)
        oldspeeds = deepcopy(newspeeds)
        oldtorques = deepcopy(newtorques)
        oldindicies = deepcopy(newindicies)
        newts, newspeeds, newtorques,newindicies = prunefastspeedchangeonce(oldts, oldspeeds, oldtorques, oldindicies; delta = delta)
    end
    return newts, newspeeds, newtorques,newindicies
end


function gettorquedatafromwav(filename; segmentlength = 0.5, segmentfrequency = 0.125)
    pts, fs = wavread(filename)
    nptsinsegment = Int64(floor(segmentlength*fs/2))*2
    nptsbetweensegments = Int64(floor(segmentfrequency*fs))
    nsegments = Int64(floor((length(pts) - (2*nptsinsegment))/nptsbetweensegments))
    middleindicies = collect(nptsinsegment:nptsbetweensegments:(length(pts)-nptsinsegment))
    bottomindicies = middleindicies .- Int64(floor(nptsinsegment/2))
    topindicies = middleindicies .+ nptsinsegment .- Int64(floor(nptsinsegment/2)+1)
    ts = (1/fs) * middleindicies
    speeds = similar(ts)
    torques = similar(ts)
    center = meanvec(pts)
    #Threads.@threads 
    for (i,t) in enumerate(ts)
        firstindex = bottomindicies[i]
        lastindex = topindicies[i]
        segpts = pts[firstindex:lastindex]
        deltat = (1/fs) * (lastindex-firstindex)
        kmax = Int64(floor(666.7 * deltat))#20k RPM
        segfft = abs.(fft(segpts))[1:kmax]
        k = 1
        findfactor = 0.1
        newspeed = 0.0
        while (newspeed < 4000.0) & (findfactor <= 1.0)
            k = findfirst(x->(x >= 0.1*maximum(segfft)),segfft)
            newspeed = 60*((k-1)/deltat)*(1/2)
            findfactor = findfactor + 0.01
        end
        speeds[i] = newspeed
        torques[i] = meanvec(abs.(segpts .- center))*1.5#*Nm_per_one#segfft[k]*Nm_per_one
        #println((torques[i],segfft[k]))
    end
    return ts,speeds,torques
end

function gettorquecalibrationfromwav(filename)
    pts, fs = wavread(filename)
    center = meanvec(pts)
    return meanvec(abs.(pts .- center))*1.5
end

function readmotorcsv(filename)
    voltages = (CSV.File(filename; header = false) |> Tables.matrix)[1,:] #CSV is one row of data
    ts = (1.0/5000.0)*collect(0:1:length(voltages))
    return ts,voltages
end

function gettorquedatafromcsv(filename,Nm_per_V; segmentlength = 0.5, segmentfrequency = 0.125)
    tpts, ypts = readmotorcsv(filename)
    nptsinsegment = Int64(floor(segmentlength*5000))#The DAQ is always 5kHz
    nptsbetweensegments = Int64(floor(segmentfrequency*5000))
    nsegments = Int64(floor((length(ypts) - (2*nptsinsegment))/nptsbetweensegments))
    middleindicies = collect(nptsinsegment:nptsbetweensegments:(length(ypts)-nptsinsegment))
    bottomindicies = middleindicies .- Int64(floor(nptsinsegment/2))
    topindicies = middleindicies .+ nptsinsegment .- Int64(floor(nptsinsegment/2)+1)
    ts = (1/5000) * middleindicies
    speeds = similar(ts)
    torques = similar(ts)
    center = meanvec(ypts)
    #Threads.@threads 
    for (i,t) in enumerate(ts)
        firstindex = bottomindicies[i]
        lastindex = topindicies[i]
        segpts = pts[firstindex:lastindex]
        torques[i] = meanvec(abs.(segpts .- center))*1.5*Nm_per_V
    end
    return ts, torques # these are returning empty
end

function getsounddatafromwav(filename; segmentlength = 0.5, segmentfrequency = 0.125)
    pts, fs = wavread(filename)
    nptsinsegment = Int64(floor(segmentlength*fs/2))*2
    nptsbetweensegments = Int64(floor(segmentfrequency*fs))
    nsegments = Int64(floor((length(pts) - (2*nptsinsegment))/nptsbetweensegments))
    middleindicies = collect(nptsinsegment:nptsbetweensegments:(length(pts)-nptsinsegment))
    bottomindicies = middleindicies .- Int64(floor(nptsinsegment/2))
    topindicies = middleindicies .+ nptsinsegment .- Int64(floor(nptsinsegment/2)+1)
    ts = (1/fs) * middleindicies
    frequencies_fft = collect(0:1:(nptsinsegment/2 - 1))* (fs/nptsinsegment)
    spectra_fft = zeros(length(ts),length(frequencies_fft))
    #Threads.@threads 
    for (i,t) in enumerate(ts)
        firstindex = bottomindicies[i]
        lastindex = topindicies[i]
        segpts = pts[firstindex:lastindex]
        spectra_fft[i,:] = (abs.(fft(segpts)))[1:Int64(length(segpts)/2)]#TODO: the intensities should be normalized into power spectral density
    end
    frequencybinlims = vcat([0],collect(1:2:99),exp10.(range(log10(101), stop=log10(48010), length=200)))#adjust number of bins if GPR is fitting too slowly.
    binnedspectra = zeros(length(ts),length(frequencybinlims)-1)
    frequencies_fft_index = 1
    push!(frequencies_fft, 2*frequencybinlims[end])#prevents a bug in the while loop where frequencies_fft is indexed beyond its size
    for i = 2:1:length(frequencybinlims)
        summedamplitudes = zeros(length(ts))
        numberfrequenciesinbin = 0
        while frequencies_fft[frequencies_fft_index] < frequencybinlims[i]
            summedamplitudes = summedamplitudes .+ spectra_fft[:,frequencies_fft_index]
            frequencies_fft_index = frequencies_fft_index + 1
            numberfrequenciesinbin = numberfrequenciesinbin + 1
        end
        if numberfrequenciesinbin > 0
            binnedspectra[:,i-1] = summedamplitudes ./ numberfrequenciesinbin
        end
    end
    return frequencybinlims, binnedspectra
end

function getpressure(prop)
    if (prop == "5") || (prop == "7")
        return 85337.0, 3070.0 # 88000, 2800
    else
        return 85337.0, 3070.0
    end
end

function cavitationnumberfromRPM(speeds;prop = "1")
    pressure_a, pressure_v = getpressure(prop)
    m_advance_per_rev = 0.0381
    if (prop == "2") || (prop == "6")
        m_advance_per_rev = 0.025401
    end
    tip_speeds = sqrt.((speeds*m_advance_per_rev/60) .^ 2 .+ (speeds*0.079796/60) .^ 2)
    return @. (pressure_a - pressure_v)/(0.5*997.45*(tip_speeds^2))
end

function getpropaudiodata(prop::String)
    Nm_per_oneamplitude = .25412932  #just get this conversion factor manually from what you recorded on LABview divided by the wavtorque in that same section
    tsmotor, speedsraw, torquesraw = gettorquedatafromwav("DATA/hydrophone/raw/DATASET1_"*prop*"/motor.wav"; segmentlength = 0.5, segmentfrequency = 0.125)
    frequencybinlims, spectrabins = getsounddatafromwav("DATA/hydrophone/raw/DATASET1_"*prop*"/hydrophone.wav"; segmentlength = 0.5, segmentfrequency = 0.125)
    ts, speeds, torques, indicies = prunefastspeedchange(prunefastspeedchangeonce(pruneslow(tsmotor, speedsraw, torquesraw)...; delta = 400.0)...;  delta = 900.0)
    torques = torques*Nm_per_oneamplitude
    return ts[1:end-1], speeds[1:end-1], torques[1:end-1], frequencybinlims, spectrabins[indicies[1:end-1],:]
end

function torqueconstantsfromtorques(torques,speeds,normalizationtorquefromspeed)
    torqueconstants = similar(torques)
    for i = 1:1:length(torques)
        normalizationtorque = normalizationtorquefromspeed(speeds[i])#TODO: this has some uncertainty, maybe add a way to account for it.
        realtorque = torques[i] - normalizationtorque
        torqueconstants[i] = realtorque/(997.45*((speeds[i]/60)^2)*(0.0254^5))
    end
    return torqueconstants
end

function downsample(speeds_full,torques_full,spectra_full)#TODO: seems to downsample too much. (to 60 instead of 500)
    similaritythreshold = 0.01
    duplicationthreshold = 0.01
    targetlength = 500
    indicies = []
    indicies_temp = []
    speeds_temp = [-1000.0]
    n_ittr = Int64(floor(length(speeds_full)/5))
    for i = 1:1:n_ittr
        range_five = ((n_ittr-1)*5+1):(n_ittr*5)
        five_speeds = speeds_full[range_five]
        avg_speed = meanvec(five_speeds)
        if (maximum(five_speeds) < ((1+similaritythreshold)*avg_speed)) && (minimum(five_speeds) > ((1-similaritythreshold)*avg_speed))
            nsimilar = sum(Int64((x > (avg_speed-100))&&(x < (avg_speed+100))) for x in speeds_temp)
            if (nsimilar/length(speeds_full)) < duplicationthreshold
                append!(indicies_temp,collect(range_five))
                append!(speeds_temp,five_speeds)
            end
        end
    end
    if length(indicies_temp) >= targetlength
        jump = length(indicies_temp)/targetlength
        i = 0
        jump_tracker = 0
        while i <= length(indicies_temp)
            jump_tracker = jump + jump_tracker
            while jump_tracker >= 1
                i = i + 1
                jump_tracker = jump_tracker - 1
            end
            push!(indicies,indicies_temp[i])
        end
    else
        indicies = indicies_temp
    end
    speeds = zeros(length(indicies))
    torques = zeros(length(indicies))
    spectra = zeros(length(indicies),size(spectra_full)[2])
    for (j,i) in enumerate(indicies)
        speeds[j] = speeds_full[i]
        torques[j] = torques_full[i]
        spectra[j,:] = spectra_full[i,:]
    end
    return speeds, torques, spectra
end

function fittorqueandspectrumtospeed(speeds_full, torques_full, spectra_full)
    speeds, torques, spectra = downsample(speeds_full, torques_full, spectra_full)
    speeds .+= 3*pi * randn(length(speeds))
    
    # Check for NaNs or Infs in speeds, torques, and spectra
    if any(isnan, speeds) || any(isinf, abs.(speeds))
        error("Found NaN or Inf in speeds.")
    end
    if any(isnan, torques) || any(isinf, abs.(torques))
        error("Found NaN or Inf in torques.")
    end
    if any(isnan, spectra) || any(isinf, abs.(spectra))
        error("Found NaN or Inf in spectra.")
    end

    println("Downsampled spectra size: ", size(spectra))

    # Set up kernel and log-noise (jitter to avoid PosDef errors)
    kernel = SEIso(log(1.0), log(0.1))
    log_noise_torque = log((0.001*meanvec(torques))^2)#adjust these values to tweak the fit.
    log_noise_spectra = log(1e-1)

    # GP for torque
    gp_torque = GP(Float64.(speeds), Float64.(torques), MeanZero(), kernel, log_noise_torque)
    optimize!(gp_torque; noise=false)

    # GPs for each frequency
    gp_spectra = Vector{GPE}(undef, size(spectra, 2))
    for i in 1:size(spectra, 2)
        y = Float64.(spectra[:, i])
        gp = GP(Float64.(speeds), y, MeanZero(), kernel, log_noise_spectra)
        optimize!(gp; noise=false)
        gp_spectra[i] = gp
    end

    # GP evaluation closures
    function torquefromspeed(speed)
        torque, _ = predict_y(gp_torque, [Float64(speed)])
        return torque[1]
    end

    function spectrumfromspeed(speed)
        spectrumout = similar(spectra[1, :])
        for i in 1:length(spectrumout)
            prediction, _ = predict_y(gp_spectra[i], [Float64(speed)])
            spectrumout[i] = prediction[1]
        end
        return spectrumout
    end

    return torquefromspeed, spectrumfromspeed
end

#=
function fittorqueandspectrumtospeed(speeds_full,torques_full,spectra_full)
    speeds,torques,spectra = downsample(speeds_full,torques_full,spectra_full)
    println(size(spectra))
    kernel = SEIso(log(1.0), log(0.1))#SE(1.0, 1.0)
    gp_torque = GP(Float64.(speeds), Float64.(torques), MeanZero(), kernel, 1e-3)
    optimize!(gp_torque; noise = true)
    gp_spectra = []
    frequenciesinspectrum = size(spectra)[2]
    for i = 1:1:frequenciesinspectrum
        gp_thisfrequency = GP(Float64.(speeds), Float64.(spectra[:,i]), MeanZero(), kernel, 1e-3)
        optimize!(gp_thisfrequency; noise = true)
        push!(gp_spectra,gp_thisfrequency)
    end
    function torquefromspeed(speed)
        torque, uncertainty = predict_y(gp_torque, [Float64(speed)])
        return torque[1]
    end
    function spectrumfromspeed(speed)
        spectrumout = similar(spectra[1,:])
        uncertaintyout = similar(spectra[1,:])
        for i = 1:1:length(spectrumout)
            temp,uncertaintyout[i] = predict_y(gp_spectra[i], [Float64(speed)])
            spectrumout[i] = temp[1]
        end
        return spectrumout
    end
    return torquefromspeed, spectrumfromspeed
end =#



function analyzeaudio()
    props = ["1" "2" "3" "4" "5" "6" "7" "8"]
    slow = ["2" "6"]
    temp1, normalizationspeedsslow, normalizationtorquesslow, temp2, normalizationspectraslow = getpropaudiodata("0_1")
    temp1, normalizationspeedsfast, normalizationtorquesfast, temp2, normalizationspectrafast = getpropaudiodata("0_2")
    #println(size(normalizationspectrafast))
    #println(length(normalizationtorquesfast))
    slownormalizationtorquefromspeed, slownormalizationspectrumfromspeed = fittorqueandspectrumtospeed(Float64.(normalizationspeedsslow),Float64.(normalizationtorquesslow),Float64.(normalizationspectraslow))
    fastnormalizationtorquefromspeed, fastnormalizationspectrumfromspeed = fittorqueandspectrumtospeed(Float64.(normalizationspeedsfast),Float64.(normalizationtorquesfast),Float64.(normalizationspectrafast))
    for i = 1:1:length(props)
        println(props[i])
        ts, speeds, torques, frequencies, spectra = getpropaudiodata(props[i])
        cavitationnumbers = cavitationnumberfromRPM.(speeds;prop = props[i])
        torqueconstants = similar(cavitationnumbers)
        if contains(slow,props[i])
            torqueconstants = torqueconstantsfromtorques(torques,speeds,slownormalizationtorquefromspeed)
        else
            torqueconstants = torqueconstantsfromtorques(torques,speeds,fastnormalizationtorquefromspeed)
        end
        #TODO: process spectra with normalization data
        #TODO: add a plot making function
        prop = parse(Int64, props[i])
        #p =plot(speeds,torques)
        #scatter!(speeds,torques)
        p =plot(cavitationnumbers,torqueconstants; xscale = :log10)
        savefig(p,"DATA/output/figures/torque_v_cavitation_$prop"*".png")
    end
end

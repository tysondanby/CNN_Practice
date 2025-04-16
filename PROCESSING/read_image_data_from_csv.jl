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
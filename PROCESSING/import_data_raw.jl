using Gtk, CSV, DataFrames

function getdatasetID()
    datasetIDcsv=CSV.File("DATA/nextdatasetID.csv")
    datasetID = datasetIDcsv.ID[1]
    datasetIDcsv.ID[1] = datasetIDcsv.ID[1] + 1
    CSV.write("DATA/nextdatasetID.csv",datasetIDcsv)
    return datasetID
end

function datestringfromfilename(filename)
    firstunderscoreindex = findfirst(x -> x == '_',filename)
    if firstunderscoreindex == 2
        return "0"*filename[1:9]
    elseif firstunderscoreindex == 3
        return filename[1:10]
    end
end

function propnumandspeedsfromfilename(filename)
    lastspaceindex = findlast(x -> x == ' ',filename)
    periodindex = findlast(x -> x == '.',filename)
    propnumandspeeds = filename[lastspaceindex+1:periodindex-1]
    if !contains(propnumandspeeds,'_')
        propnumandspeeds = propnumandspeeds*"_4500_15000"
    end
    return propnumandspeeds
end

function propnumandspeedsfromfoldername(foldername)
    propnumandspeeds = foldername
    if !contains(propnumandspeeds,'_')
        propnumandspeeds = propnumandspeeds*"_4500_15000"
    end
    return propnumandspeeds
end

function readweathersanddays(dir)#dir should be a folder with a raw dataset
    weathertemp = CSV.File(dir*"/weather.csv")
    daystemp = CSV.File(dir*"/days.csv")
    datasetnames = daystemp.Dataset
    dates = daystemp.Date
    pressures = zeros(length(dates))
    temperatures = zeros(length(dates))
    speedmaxs = zeros(length(dates))
    speedmins = zeros(length(dates))
    propellers = zeros(length(dates))
    for i = 1:1:length(dates)
        weatherindex = findfirst(x -> x == dates[i], weathertemp.Date)
        pressures[i] = weathertemp.Pressure[weatherindex]
        temperatures[i] = weathertemp.Temperature[weatherindex]
        if !contains(datasetnames[i],'_')
            speedmaxs[i] = 15000
            speedmins[i] = 4500
            propellers[i] = parse(Int32,datasetnames[i])
        else
            firstunderscoreindex = findfirst(x -> x == '_',datasetnames[i])
            lastunderscoreindex = findlast(x -> x == '_',datasetnames[i])
            propellers[i] = parse(Int32,datasetnames[i][1:firstunderscoreindex-1])
            speedmins[i] = parse(Int32,datasetnames[i][firstunderscoreindex+1:lastunderscoreindex-1])
            speedmaxs[i] = parse(Int32,datasetnames[i][lastunderscoreindex+1:end])
        end
        
    end
    return datasetnames, dates, pressures, temperatures, propellers, speedmaxs, speedmins
end

function addmetadata(metadatacsvfilename,addname, adddate, addtemp, addpressure)
    metadata = CSV.File(metadatacsvfilename)
    push!(metadata.name,addname)
    push!(metadata.date,adddate)
    push!(metadata.temp,addtemp)
    push!(metadata.pressure,addpressure)
    CSV.write(metadatacsvfilename,DataFrame(metadata))
end

function moveDAQcsv(datasetID,dir,subdir,datasetnames, dates, pressures, temperatures,propellers, speedmaxs, speedmins)
    datestring = datestringfromfilename(subdir)
    propnumandspeeds = propnumandspeedsfromfilename(subdir)
    newfoldername = "DATASET$datasetID"*'_'*propnumandspeeds
    if !isdir("DATA/hydrophone/raw/"*newfoldername)
        mkdir("DATA/hydrophone/raw/"*newfoldername)
        weatherindex = findfirst(x -> x == datestring,dates)
        addmetadata("DATA/hydrophone/raw/metadata.csv",newfoldername,datestring,temperatures[weatherindex], pressures[weatherindex])
    end
    cp(dir*"/"*subdir,"DATA/hydrophone/raw/"*newfoldername*"/DAQ.csv")
end

function movetif(datasetID,dir,subdir,datasetnames, dates, pressures, temperatures,propellers, speedmaxs, speedmins)
    weatherindex = 0
    underscoreindex = findfirst(x -> x == '_',subdir)
    lastunderscoreindex = findlast(x -> x == '_',subdir)
    propeller = parse(Int32,subdir[1:underscoreindex-1])
    speed = parse(Int32,subdir[underscoreindex+1:lastunderscoreindex-1])
    for j = 1:1:length(datasetnames)
        if ((propeller == propellers[j]) && (speed >= speedmins[j]) && (speed <= speedmaxs[j]))
            weatherindex = j
        end
    end
    addmetadata("DATA/camera/raw/metadata.csv","DATASET$datasetID"*'_'*subdir,dates[weatherindex],temperatures[weatherindex], pressures[weatherindex])
    cp(dir*"/"*subdir,"DATA/camera/raw/"*"DATASET$datasetID"*'_'*subdir)
end

function movewav(datasetID,dir,subdir,datasetnames, dates, pressures, temperatures,propellers, speedmaxs, speedmins)
    weatherindex = findfirst(x -> x == subdir,datasetnames)
    propnumandspeeds = propnumandspeedsfromfoldername(subdir)
    newfoldername = "DATASET$datasetID"*'_'*propnumandspeeds
    if !isdir("DATA/hydrophone/raw/"*newfoldername)
        mkdir("DATA/hydrophone/raw/"*newfoldername)
        addmetadata("DATA/hydrophone/raw/metadata.csv",newfoldername,dates[weatherindex],temperatures[weatherindex], pressures[weatherindex])
    end
    subsubdirs = readdir(dir*"/"*subdir)
    for subsubdir in subsubdirs
        sourcedir = dir*"/"*subdir*"/"*subsubdir
        if contains(subsubdir,"Tr1")
            cp(sourcedir,"DATA/hydrophone/raw/"*newfoldername*"/motor.WAV")
        elseif contains(subsubdir,"Tr2")
            cp(sourcedir,"DATA/hydrophone/raw/"*newfoldername*"/hydrophone.WAV")
        end
    end
end

function importnew()
    #GET DIR
    dir = open_dialog("Select Dataset Folder", action=GtkFileChooserAction.SELECT_FOLDER)
    if isdir(dir)
        datasetID = getdatasetID()
        datasetnames, dates, pressures, temperatures,propellers, speedmaxs, speedmins = readweathersanddays(dir)
        subdirs = readdir(dir)
        ndirs = length(subdirs)
        for i = 1:1:ndirs
            if (contains(subdirs[i],".csv") && !(contains(subdirs[i],"weather.csv")) && !(contains(subdirs[i],"days.csv")))
                moveDAQcsv(datasetID,dir,subdirs[i],datasetnames, dates, pressures, temperatures,propellers, speedmaxs, speedmins)
            elseif contains(subdirs[i],".tif")
                movetif(datasetID,dir,subdirs[i],datasetnames, dates, pressures, temperatures,propellers, speedmaxs, speedmins)
            elseif !contains(subdirs[i],".")
                movewav(datasetID,dir,subdirs[i],datasetnames, dates, pressures, temperatures,propellers, speedmaxs, speedmins)
            end
            println("Moved $i / $ndirs")
        end
    end
end

importnew()
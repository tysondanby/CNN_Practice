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

function gettorquecalibrationfromwav(filename)#run from REPL. not called in program.
    pts, fs = wavread(filename)
    center = meanvec(pts)
    return meanvec(abs.(pts .- center))*1.5
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

function getpropaudiodata(prop::String)
    Nm_per_oneamplitude = .25412932  #just get this conversion factor manually from what you recorded on LABview divided by the wavtorque in that same section
    tsmotor, speedsraw, torquesraw = gettorquedatafromwav("DATA/hydrophone/raw/DATASET1_"*prop*"/motor.wav"; segmentlength = 0.5, segmentfrequency = 0.125)
    frequencybinlims, spectrabins = getsounddatafromwav("DATA/hydrophone/raw/DATASET1_"*prop*"/hydrophone.wav"; segmentlength = 0.5, segmentfrequency = 0.125)
    ts, speeds, torques, indicies = prunefastspeedchange(prunefastspeedchangeonce(pruneslow(tsmotor, speedsraw, torquesraw)...; delta = 400.0)...;  delta = 900.0)
    torques = torques*Nm_per_oneamplitude
    return ts[1:end-1], speeds[1:end-1], torques[1:end-1], frequencybinlims, spectrabins[indicies[1:end-1],:]
end
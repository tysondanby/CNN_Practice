function downsample_old(speeds_full,torques_full,spectra_full)#TODO: seems to downsample too much. (to 60 instead of 500)
    similaritythreshold = 0.1
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
            nsimilar = sum(Int64((x > (avg_speed-20))&&(x < (avg_speed+20))) for x in speeds_temp)
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

function downsample(speeds_full,torques_full,spectra_full)#TODO: seems to downsample too much. (to 60 instead of 500)
    similaritythreshold = 0.1
    duplicationthreshold = 0.01
    targetlength = 500
    indicies = []
    indicies_temp = collect(1:1:length(speeds_full))
    speeds_temp = [-1000.0]
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
            if i <= length(indicies_temp)
                push!(indicies,indicies_temp[i])
            end
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
    speeds, torques, spectra = downsample(speeds_full, torques_full, spectra_full) #TODO (speeds_full, torques_full, spectra_full)#
    speeds .+= (3*pi * randn(length(speeds))) .- (1.5*pi) #works at 6 and 3
    
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
    log_noise_torque = log((0.001*meanvec(torques))^2)#TODO adjust these values to tweak the fit. make them as small as possible
    log_noise_spectra = log(1e0)

    # GP for torque
    gp_torque = GP(Float64.(speeds), Float64.(torques), MeanZero(), kernel, log_noise_torque)
    optimize!(gp_torque; noise=false)

    # GPs for each frequency
    gp_spectra = Vector{GPE}(undef, size(spectra, 2))
    for i in 1:size(spectra, 2)
        y = Float64.(spectra[:, i])
        log_noise_spectra = log((0.001*meanvec(y))^2) #try to debug
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
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

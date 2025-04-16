function cavitationnumberfromRPM(speeds;prop = "1")
    pressure_a, pressure_v = getpressure(prop)
    m_advance_per_rev = 0.0381
    if (prop == "2") || (prop == "6")
        m_advance_per_rev = 0.025401
    end
    tip_speeds = sqrt.((speeds*m_advance_per_rev/60) .^ 2 .+ (speeds*0.079796/60) .^ 2)
    return @. (pressure_a - pressure_v)/(0.5*997.45*(tip_speeds^2))
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
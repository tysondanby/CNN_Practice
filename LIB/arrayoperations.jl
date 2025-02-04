import Base.contains

function selectrandom(list)
    index = Int32(round(rand()*(length(list)-1)) +1)
    return list[index]
end
function contains(a::Matrix{T1}, e::T2) where T1 <: Any where T2 <: Any
    for element in a
        if element == e
            return true
        end
    end
    return false
end
using AutoHashEquals

@auto_hash_equals struct Weight{T<:AbstractFloat}
    weight::T
    from::Node
    to::Node
end

weight(w::Weight) = w.weight
from(w::Weight) = w.from
to(w::Weight) = w.to

function set_weight!(w::Weight, weight)
    w.weight = weight
end
function set_from!(w::Weight, from)
    w.from = from
end
function set_to!(w::Weight, to)
    w.to = to
end
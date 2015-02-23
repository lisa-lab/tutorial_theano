# The weird thing is that the function succeeds.
#
# This is weird because the two values passed in for x and y do not
# have the same shape, yet x is added with something that has the same
# shape as y (z).
#
# This happens because optimizations realize that z is always zero and
# therefore remove the addition, which removes the error.
#
# The problem is more evident if FAST_COMPILE or DEBUG_MODE is used.

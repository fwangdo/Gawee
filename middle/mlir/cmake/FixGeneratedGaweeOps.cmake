if(NOT DEFINED GAWEE_OPS_CPP)
  message(FATAL_ERROR "GAWEE_OPS_CPP must be set")
endif()

file(READ "${GAWEE_OPS_CPP}" gawee_ops_cpp)
string(REPLACE "\"requires attribute" "\" \"requires attribute" gawee_ops_cpp "${gawee_ops_cpp}")
file(WRITE "${GAWEE_OPS_CPP}" "${gawee_ops_cpp}")

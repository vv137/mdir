file(READ "${INPUT}" bytes HEX)
string(REGEX REPLACE "(..)" "0x\\1," bytes "${bytes}")
file(WRITE "${OUTPUT}" "static const unsigned char pmeCallbackCode[] = {${bytes}};\n")

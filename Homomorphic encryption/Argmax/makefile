## TFHE LIBRARY
TFHE_FOLDER_PATH = <Path to the TFHE library folder>
TFHE_BUILD_FOLDER = <The name of the TFHE build folder>

TFHE_LIB_PATH = $(TFHE_FOLDER_PATH)/$(TFHE_BUILD_FOLDER)/libtfhe
TFHE_INCLUDE_PATH = $(TFHE_FOLDER_PATH)/src/include

CC := g++ -std=c++14

# LOCAL paths
IDIR := ../include
ODIR := ../obj
BDIR := ../bin
DDIR := ../data

INCS:= -I${TFHE_INCLUDE_PATH} -I${IDIR}

LIBS:= -L${TFHE_LIB_PATH} -ltfhe-spqlios-avx

# Executable objects
_OBJ_TEST = argmax.o
OBJ_TEST = $(patsubst %,$(ODIR)/%,$(_OBJ_TEST))

.PHONY := all clean delete

all: $(BDIR)/test

$(BDIR)/test: $(OBJ_TEST)
	$(CC) $^ -o $@ ${INCS} $(LIBS)

clean:
	rm -f $(ODIR)/*.o $(BDIR)/*

delete:
	rm -f $(DDIR)/*

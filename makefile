NVCC =nvcc
CC=gcc

# CUDAFLAGS = -arch=compute_60 # Pascal (60, 61, 62)
# CUDAFLAGS = -arch=compute_70 # Volta (70, 72)
# CUDAFLAGS = -arch=compute_75 # Turing
# CUDAFLAGS = -arch=compute_80 # Ampere
# CUDAFLAGS = -arch=compute_89 # Ada
# CUDAFLAGS = -arch=compute_90 # Hopper
# CUDAFLAGS = -arch=compute_100 # Blackwell

#CUDAFLAGS = -arch=compute_60 -lineinfo -O3 -maxrregcount 24
CUDAFLAGS = -arch=compute_60 -O3

SOURCES = $(wildcard cuda/*.cu) $(wildcard cuda/test/*.cu) $(wildcard gbees/*.cu)
OBJS := $(patsubst %.cu,%.o,$(SOURCES))

TARGET = GBEES

.DEFAULT: all

all: link

link: ${OBJS}
	${NVCC} ${CUDAFLAGS} -o ${TARGET} ${OBJS}

%.o: %.cu
	${NVCC} ${CUDAFLAGS} -dc -x cu $< -o $@

clean:
	rm -f cuda/*.o 	
	rm -f cuda/test/*.o 	
	rm -f gbees/*.o 	
	rm -f ${TARGET}

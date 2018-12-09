SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=)

all: $(OBJ)

%: %.cu common.hu Makefile
	nvcc $< -arch=sm_70 -lcufftw -lcufft \
		-rdc=true -lcublas -lcublas_device -lcudadevrt \
		--use_fast_math -Xcompiler -fopenmp \
		--std=c++11 -O3 -o $@ 2>&1 | grep -v "ptxas info"

test: $(OBJ)
	for o in $(OBJ) ; do \
		echo "=============== $${o} $${stride}"; \
		./$$o;									 \
	done	

clean:
	rm -f $(OBJ)

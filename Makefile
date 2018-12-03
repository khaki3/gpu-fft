SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=)

all: $(OBJ)

%: %.cu common.hu Makefile
	nvcc -lcufftw -lcufft -Xcompiler -fopenmp --std=c++11 -O3 -o $@ $<

test: $(OBJ)
	for o in $(OBJ) ; do \
		echo "=============== $${o} $${stride}"; \
		./$$o;									 \
	done	

clean:
	rm -f $(OBJ)

main : main.cpp
	mpicxx main.cpp -o main -std=c++11 -O3

.PHONY : clean
clean :
	-rm main
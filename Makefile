

all:
	gcc -c -fPIC -lm -O3 ./src/fmen_x2_div2k_35dB.c -o fmen_x2_div2k_35dB.o
	gcc -shared fmen_x2_div2k_35dB.o -o fmen_x2_div2k_35dB.so
	rm fmen_x2_div2k_35dB.o

clean:
	rm *.so *.o

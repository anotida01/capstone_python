

all:
	gcc -c -fPIC -lm -O3 fmen_x2_div2k_35dB.c
	gcc -shared fmen_x2_div2k_35dB.o -o fmen_x2_div2k_35dB.so
	rm *.o

clean:
	rm *.so *.o

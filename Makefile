

all:
	gcc -c -fPIC -lm -O3 fmen1080_new.c
	gcc -shared fmen1080_new.o -o fmen1080_new.so

clean:
	rm *.so *.o

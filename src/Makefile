all:a.out

a.out: main.cpp FaceFeature.cpp  FaceRegion.cpp  GrabCut.cpp utility_functions.cpp
	g++  main.cpp FaceFeature.cpp  FaceRegion.cpp  GrabCut.cpp utility_functions.cpp HairRegion.cpp Accessories.cpp `pkg-config --cflags --libs opencv` -o makeover

clean:
	rm -rf a.out 

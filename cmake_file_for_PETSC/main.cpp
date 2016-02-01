#include "petscvec.h"

int main(int argc, char **argv) {
	Vec x;
	
	PetscInitialize(&argc, &argv, NULL, NULL);
	
	VecCreateSeq(PETSC_COMM_SELF, 100, &x);
	VecSet(x, 1.);
	
   
    PetscFinalize();
    return 0;
}

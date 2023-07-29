TEST_SOURCE = hello_from_gpu.cu

FINALLY_SOURCE = hello_cuda_01.cu

TARGETBIN := ./hello_cuda_01

MIDDLE_OUTPUT = hello_from_gpu.o

CC = nvcc


$(TARGETBIN):$(MIDDLE_OUTPUT) $(FINALLY_SOURCE)
	$(CC)  $(MIDDLE_OUTPUT) $(FINALLY_SOURCE) -o $(TARGETBIN)
$(MIDDLE_OUTPUT):$(TEST_SOURCE)
	$(CC) --device-c $< -o $@

# the below is the same as the above
# $(TARGETBIN):$(MIDDLE_OUTPUT) $(FINALLY_SOURCE)
#	$(CC) $^ -o $@


.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
	-rm -rf *.o
    
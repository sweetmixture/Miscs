import numpy as np
tmp = np.array([2,3,6,7,10,13,14])       # 배열 tmp를 생성
print(tmp)

test1 = np.where(tmp<10) # 배열 tmp에서 10보다 작은 원소의 index를 반환하는 배열을 생성
print(test1,type(test1))
print(test1[0])

test2 = np.where(tmp%2==0) # 배열 tmp에서 2의 배수인 원소의 index를 반환하는 배열을 생셩
print(test2[0])

test3 = np.where(tmp<4,0,1)
print(test3)
print('------------------')
# ---------------
mat = np.array([[0,1,2],[0,2,4],[0,3,6]])    # mat이라는 배열을 정의    
mat_where = np.where(mat<4, mat, -1)    # 배열 mat의 원소 중 4보다 큰 값에서는 -1을 반환
print(mat)
print(mat_where)
print()

a = np.array([1,2,3,4])
b = np.array([2,0,1,4])
test = np.where(a<b, a, b+10)  # 배열 a,b의 원소를 1:1로 비교하는데, a의 원소가 크면 a를 반환, 아니면 b+10을 반환
print(test)
print()

a1 = np.array([[2],[3],[4]])
b1 = np.array([1,2,3])
test1 = np.where(a1<b1, a1, b1)  # a1이 3 by 1 배열로 첫번째 열과 b1을 비교하여 조건에 맞게 a1 또는 b1+10을 반환
print(test1)

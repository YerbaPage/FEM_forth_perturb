a = 1
print('Original:', a)
def test(num):
    global a
    for k in range(num):
        a += 1
        print('in process:', a)

if __name__ == '__main__':
    test(2)
    print('after: ', a)
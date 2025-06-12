
def function(a_, b_):
    tmp = a_
    a_ = b_
    b_ = tmp
    print(a_, b_)



if __name__ == '__main__':
    a = 10
    b = 100
    function(a,b)
    print(a, b)

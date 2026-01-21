##----------------------------------------------------------------------------------------------------------
##==========================================================================================================
# # Write 5 functions:
##==========================================================================================================
##----------------------------------------------------------------------------------------------------------
# Sum of list

## Using for loop
s=0
for i in list_input:
    s=s+i

# Using builtin funtion sum()
def sum_of_list(list1):
    return sum(list1)
    
list_input = list(map(int,input("Enter numbers sepereated by space : ").split()))

print(sum_of_list(list_input))

##----------------------------------------------------------------------------------------------------------
#max of list (no max())
list_1 = list(map(int, input("Enter Numbers of list seperated by space :  ").split()))
max_no = max(list_1)
print(max_no)

##----------------------------------------------------------------------------------------------------------
#count vowels in string
str_v  =input("enter a word with vowels : ")
vowels = ['a','e','i','o','u']
ctr=0
for i in str_v:
    if i in vowels:
        ctr+=1
print("Total number of vowels in " ,str_v, " is ", ctr)

##----------------------------------------------------------------------------------------------------------
#check palindrome
str_v = input("enter String to check palindrome : ")
if str_v == str_v[::-1]:
    print('yes, It is a palindrome')
else:
    print("Not a palindrome")

##----------------------------------------------------------------------------------------------------------
# factorial (loop, not recursion) 5! = 1x2x3x4x5 = 120
n = int(input("Enter a number : "))
fact = 1
for i in range(n,0,-1):
    fact = fact *i
print(fact)

##----------------------------------------------------------------------------------------------------------
##==========================================================================================================
# # Data structures:
##==========================================================================================================
##----------------------------------------------------------------------------------------------------------



# Create a list, tuple, set, dict — add, remove, iterate each.
li = list(map(int,input("Enter numbers for list seperated by space : ").split()))
tu = tuple(map(str,input("Enter words for tuple seperated by space : ").split()))
se = set(li)
dict = {"a":1,"b":2}
print(li,tu,se,dict,sep="\n")
##----------------------------------------------------------------------------------------------------------

#Add to list
n1=int(input("Enter number to add to the list : "))
li.append(n1)
li.pop(1)
print(li)
##----------------------------------------------------------------------------------------------------------

#add to dict
dict2 = {"3":1,"d":2}
dict.update(dict2)
print(dict)
##----------------------------------------------------------------------------------------------------------

#Iteration
for i in li:
    print(i,i+10)
for j in tu:
    print("Hello " ,j)
for key in dict:
    print(dict[key])



##----------------------------------------------------------------------------------------------------------
#Convert list → set → list and explain why size changed.
import sys
lis = [10,20,30]
print(lis,sys.getsizeof(lis))
se = set(lis)
print(se,sys.getsizeof(se))
list_from_set = list(se)
print(list_from_set,sys.getsizeof(list_from_set))

##----------------------------------------------------------------------------------------------------------
##==========================================================================================================
# # Loops + conditionals:
##==========================================================================================================
##----------------------------------------------------------------------------------------------------------

##Print numbers 1–100 , Replace multiples of 3 → "Fizz", 5 → "Buzz", both → "FizzBuzz"
for i in range(1,1001):
    if(i%5 == 0 and i % 3 == 0):
        print("FizzBuzz")
    elif(i%3==0):
        print("Fizz")
    elif(i%5 == 0):
        print("Buzz")
    else: print(i)

###----------------------------------------------------------------------------------------------------------
###==========================================================================================================
# # # Errors:
###==========================================================================================================
###----------------------------------------------------------------------------------------------------------
try: 
    mylist = [10,20]
    mydict = {"a":1}
    print(mylist[5])
    print(mydict["b"])
except IndexError:
    print("invalid index")
except KeyError:
    print("invalid  key")
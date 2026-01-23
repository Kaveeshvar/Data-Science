###========================================
### Create a dictionary of 5 students:
###========================================

stud_dict = {
    "Kavee": {"math" : 99, "Science": 85},
    "Sriman" : { "math" : 85, "Science" :75}, 
    "Mohinth" : { "math" : 90, "Science" :80}, 
    "Anu" : { "math" : 99, "Science" :70}, 
    "Karti" : { "math" : 60, "Science" :45} 
}


###========================================
### Exercises
###========================================
### Add a new subject score
#stud_dict["Kavee"]["English"] = 77
# for v in stud_dict.values():
#     v["English"] = 95
# print(stud_dict) 

###-------------------------------------------------------

# ### Update a score
# stud_dict["Anu"]["math"] = 100
# print(stud_dict) 

###-------------------------------------------------------

### Delete a subject
# del stud_dict["Kavee"]["math"]
# stud_dict["Kavee"].pop("Science")
# print(stud_dict) 

###-------------------------------------------------------

# ### Loop and print:
# for v in stud_dict.keys():
#     avg = (stud_dict[v]["math"]+stud_dict[v]["Science"])/2
#     print(v, "Has a marks average of", avg)

###========================================
### Convert
###========================================

### List of tuples → dict
# dict_of_tup={}
# list_of_tup = [("a","b","c"),("1","2","3")]
# dict_of_tup[list_of_tup[0]] = list_of_tup[1]
# print(dict_of_tup)

###-------------------------------------------------------

### Dict → list of keys, values, items
# list_keys,list_values,list_items=[],[],[]
# for v in stud_dict.keys():
#     list_keys.append(v)
#     list_values.append(stud_dict[v])
# for k in stud_dict.items():
#     list_items.append(k)

###-------------------------------------------------------

# print(list_keys,list_values,list_items,sep="\n")
### Optimised approach
# list_keys = list(stud_dict.keys())
# list_values = list(stud_dict.values())
# list_items = list(stud_dict.items())
# print(list_keys,list_values,list_items,sep="\n")


###========================================
### Write 3 functions
###========================================

### get_average(student_dict)
# def get_average(student_dict):
#     for v in student_dict.keys():
#         avg = (student_dict[v]["math"]+student_dict[v]["Science"])/2
#         print(avg)
# get_average(stud_dict)

###-------------------------------------------------------

### get_topper(student_dict)
# def get_topper(student_dict):
#     avg = []
#     roll=0
#     for v in student_dict.keys():
#         avg.append((student_dict[v]["math"]+student_dict[v]["Science"])/2)
#     return max(avg) 
# print(get_topper(stud_dict))
# list_keys = list(stud_dict.keys())
# print(list_keys[0])

# ### Optimised
# """
#     sum(student_dict[s].values()) calculates total marks per student.

#     max() finds the student with the highest total score.

#     key tells max() to compare based on total marks, not names.
# """

# def get_topper(student_dict):
#         return max(student_dict,key = lambda s:sum(student_dict[s].values()))
# print(get_topper(stud_dict))

###-------------------------------------------------------

# ### add_student(main_dict, name, scores_dict)
# def add_student(main_dict, name, scores_dict):
#     main_dict[name] = scores_dict

# add_student(stud_dict,"Kiran",{"math" : 99, "Science": 85})
# print(stud_dict)


###========================================
### Real-world mini task:
###========================================

# ### Given a list of (movie, rating) tuples → build a dict {movie: avg_rating}
# movie_ratings = [
#     ("Kong",9),("Godzilla",9.2),("Kong",8.5),("Godzilla",8.1),
#     ("Kong",9.2),("Godzilla",6.9),("Kong",5.8),("Godzilla",8.7),
#     ("Kong",9.7),("Godzilla",9.9),("Kong",9.2),("Godzilla",7.5),
#     ]

# def movie_avg_ratings(movie_list):
#     d = {}
#     for movie, rating in movie_list:
#         d.setdefault(movie, []).append(rating)
#     return {m: sum(r)/len(r) for m, r in d.items()}
# print(movie_avg_ratings(movie_ratings))

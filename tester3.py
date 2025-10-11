# Based on testing harness dated 2017-06-02.

# STUDENTS: TO USE:
# 
# The following command will test all test cases on your file:
# 
#   python3 <thisfile.py> <your_one_file.py>
# 
# 
# You can also limit the tester to only the functions you want tested.
# Just add as many functions as you want tested on to the command line at the end.
# Example: to only run tests associated with func1 and func2, run this command:
# 
#   python3 <thisfile.py> <your_one_file.py> func1 func2
# 
# You really don't need to read the file any further, except that when
# a specific test fails, you'll get a line number - and it's certainly
# worth looking at those areas for details on what's being checked. This would
# all be the indented block of code starting with "class AllTests".


# INSTRUCTOR: TO PREPARE:
#  - add test cases to class AllTests. The test case functions' names must
# be precise - to test a function named foobar, the test must be named "test_foobar_#"
# where # may be any digits at the end, such as "test_foobar_13".
# - any extra-credit tests must be named "test_extra_credit_foobar_#"
# 
# - name all required definitions in REQUIRED_DEFNS, and all extra credit functions
#   in EXTRA_CREDIT_DEFNS. Do not include any unofficial helper functions. If you want
#   to make helper definitions to use while testing, those can also be added there for
#   clarity.
# 
# - to run on either a single file or all .py files in a folder (recursively):
#   python3 <thisfile.py> <your_one_file.py>
#   python3 <thisfile.py> <dir_of_files>
#   python3 <thisfile.py> .                    # current directory
# 
# A work in progress by Mark Snyder, Oct. 2015.
#  Edited by Yutao Zhong, Spring 2016.
#  Edited by Raven Russell, Spring 2017.
#  Edited by Mark Snyder, June 2017.


import unittest
import shutil
import sys
import os
import time


############################################################################
############################################################################
# BEGIN SPECIALIZATION SECTION (the only part you need to modify beyond 
# adding new test cases).

# name all expected definitions; if present, their definition (with correct
# number of arguments) will be used; if not, a decoy complainer function
# will be used, and all tests on that function should fail.
    
REQUIRED_DEFNS = [
                    'add_em_up', 'add_em_up2', 'zoom_boom', 'elem_loc', 'update_credits', 'clean_credits_list'
                 ]

# for method names in classes that will be tested
SUB_DEFNS = [
                
            ]

# definitions that are used for extra credit
EXTRA_CREDIT_DEFNS = []

# how many points are test cases worth?
weight_required = 1
weight_extra_credit = 0

# don't count extra credit; usually 100% if this is graded entirely by tests.
# it's up to you the instructor to do the math and add this up!
# TODO: auto-calculate this based on all possible tests.
total_points_from_tests = 45

# how many seconds to wait between batch-mode gradings? 
# ideally we could enforce python to wait to open or import
# files when the system is ready but we've got a communication
# gap going on.
DELAY_OF_SHAME = 1


# set it to true when you run batch mode... 
CURRENTLY_GRADING = False



# what temporary file name should be used for the student?
# This can't be changed without hardcoding imports below, sorry.
# That's kind of the whole gimmick here that lets us import from
# the command-line argument without having to qualify the names.
RENAMED_FILE = "student"




# END SPECIALIZATION SECTION
############################################################################
############################################################################


# enter batch mode by giving a directory to work on as the only argument.
BATCH_MODE = len(sys.argv)==2 and (sys.argv[1] in ["."] or os.path.isdir(sys.argv[1]))

# This class contains multiple "unit tests" that each check
# various inputs to specific functions, checking that we get
# the correct behavior (output value) from completing the call.
class AllTests (unittest.TestCase):

############################################################################
    def test_add_em_up_1(self):
        self.assertEqual(add_em_up(8, 10), 1)
    
    def test_add_em_up_2(self):
        self.assertEqual(add_em_up(1, 30), 7)
    
    def test_add_em_up_3(self):
        self.assertEqual(add_em_up(50, 100), 9)
    
    def test_add_em_up_4(self):
        self.assertEqual(add_em_up(22, 543), 31)
    
    def test_add_em_up_5(self):
        self.assertEqual(add_em_up(1, 1000), 44)
    
    def test_add_em_up_6(self):
        self.assertEqual(add_em_up(60, 60), 0)
    
    def test_add_em_up_7(self):
        self.assertEqual(add_em_up(1, 8), 3)
    
    def test_add_em_up_8(self):
        self.assertEqual(add_em_up(21, 3999), 88)
    
    def test_add_em_up_9(self):
        self.assertEqual(add_em_up(500, 10001), 137)

############################################################################
    def test_add_em_up2_1(self):
        print("Testing with 10, 8")
        self.assertEqual(add_em_up2(10, 8), 1)

    def test_add_em_up2_2(self):
        print("Testing with 100, 50")
        self.assertEqual(add_em_up2(100,50), 9)
    
    def test_add_em_up2_3(self):
        self.assertEqual(add_em_up2(543,22), 31)
    
    def test_add_em_up2_4(self):
        self.assertEqual(add_em_up2(1, 1000), 44)
    
###########################################################################
    def test_zoom_boom_1(self):
        ls = [5, 6, 17, 21, 77]
        print(f"Testing with {ls} and 7")
        res = [5, 6, 'Boom', 'Zoom','Zoom']
        self.assertEqual(zoom_boom(ls,7), res)
    
    def test_zoom_boom_2(self):
        ls = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 24]
        print(f"Testing with {ls} and 2") 
        res =['Zoom',25,'Zoom',27,'Zoom',29,'Zoom',31,'Zoom',33,'Zoom']
        self.assertEqual(zoom_boom(ls, 2), res)
    
    def test_zoom_boom_3(self):
        ls = [81, 82, 83, 84, 85, 86, 87, 88]
        res = [81,82,83,84,85,86,87,'Zoom']
        self.assertEqual(zoom_boom(ls, 8), res)
    
    def test_zoom_boom_4(self):
        self.assertEqual(zoom_boom(list(range(1,2)),1), ['Zoom'])
    
    def test_zoom_boom_5(self):
        ls = list(range(20))        
        self.assertEqual(zoom_boom(list(range(1,6)),3), [1, 2, 'Zoom', 4, 5])
    
    def test_zoom_boom_6(self):
        res = [1, 'Zoom', 3, 'Zoom', 5, 'Zoom', 7, 'Zoom', 9]
        self.assertEqual(zoom_boom(list(range(1,10)),2), res)
    
    def test_zoom_boom_7(self):
        res = [50, 51, 'Zoom', 53, 'Boom', 55, 'Zoom', 57, 58, 59,
               'Zoom', 61, 62, 63, 'Zoom', 65, 66, 67, 'Zoom', 69,
               70, 71, 'Zoom', 73, 'Boom', 75, 'Zoom', 77, 78, 79]
        self.assertEqual(zoom_boom(list(range(50,80)),4), res)
    
    def test_zoom_boom_8(self):
        res = [20, 21, 22, 23, 24, 25, 26, 'Zoom', 28, 'Boom', 30]
        self.assertEqual(zoom_boom(list(range(20,31)),9), res)
    
    def test_zoom_boom_9(self):
        self.assertEqual(zoom_boom(list(range(12,18)),9), [12, 13, 14, 15, 16, 17])
        
    ############################################################################
    def test_elem_loc_1(self):
        self.assertEqual(elem_loc([4,5,6,7,8,9], 6), 2)
    
    def test_elem_loc_2(self):
        self.assertEqual(elem_loc([4,5,6,7,8,9], 4), 0)
    
    def test_elem_loc_3(self):
        self.assertEqual(elem_loc([1,2,3,4,1,2,3,4,5], 4), 3)
    
    def test_elem_loc_4(self):
        self.assertEqual(elem_loc(['X', 'Y','Z'], 'M'),None)
    
    ############################################################################
    def test_update_credits_1(self):
        stu_list1=['10145',60,'10947',12,'10444',88,'10029',31,'10974',63]
        up = ['10444', 12, '10622', 24]
        result = ['10145',60,'10947',12,'10444',100,'10029',31,'10974',63,'10622',24]
        self.assertEqual(update_credits(stu_list1, up), result)
    
    def test_update_credits_2(self):
        stu_list1=['10145',60,'10947',12,'10444',88,'10029',31,'10974',63]
        up = ['10974',10,'10145',5,'10947',5]
        res = ['10145',65,'10947',17,'10444',88,'10029',31,'10974',73]
        self.assertEqual(update_credits(stu_list1, up), res)
    
    def test_update_credits_3(self):
        stu_list1=['10145',60,'10974',63]
        up = ['10947',12,'10444',88,'10029',31]
        result = ['10145',60,'10974',63,'10947',12,'10444',88,'10029',31]
        self.assertEqual(update_credits(stu_list1, up), result)
    
    def test_update_credits_4(self):
        stu_list2 = ['10537', 44, '10256', 88, '10234', 80, '10038', 104]
        up = []
        self.assertEqual(update_credits(stu_list2,up), stu_list2)
    
    def test_update_credits_5(self):
        self.assertEqual(update_credits([], ['20442',16]),['20442',16])
    
    def test_update_credits_6(self):
        stu_list2 = ['10537', 44, '10256', 88, '10234', 80, '10038', 104]
        up = []        
        self.assertEqual(update_credits(up, stu_list2), stu_list2)
    
    def test_update_credits_7(self):
        stu_list2 = ['10537', 44, '10256', 88, '10234', 80, '10038', 104]
        up = ['10537',10,'10256',10,'10038',10]
        result = ['10537',54,'10256',98,'10234',80,'10038',114]
        self.assertEqual(update_credits(stu_list2, up), result)
    
    def test_update_credits_8(self):
        stu_list2 = ['10537', 44, '10256', 88, '10234', 80, '10038', 104]
        result = ['10537', 88, '10256', 176, '10234', 160, '10038', 208]
        self.assertEqual(update_credits(stu_list2, stu_list2), result)
    
    def test_update_credits_9(self):
        self.assertEqual(update_credits([], []), [])
    ############################################################################
    def test_clean_credits_list_1(self):
        mess1 = [['10444',3],['10445',4],['10456',5],['10328',5],['12266',4]]
        result = ['10444',3,'10445',4,'10456',5,'10328',5,'12266',4]
        self.assertEqual(clean_credits_list(mess1), result)
    
    def test_clean_credits_list_2(self):
        mess = [['10444',3], ['10445',4], ['10456',5], ['10328',5], ['12266',4], ['10444',5], ['10445',5], ['10456',5], ['10328',5], ['12266',5]]
        result = ['10444', 8, '10445', 9, '10456', 10, '10328', 10, '12266', 9]
        self.assertEqual(clean_credits_list(mess), result)
    
    def test_clean_credits_list_3(self):
        mess = [['10444', 3], ['10445',4], ['10456',5], ['10328',5], ['12266',4], ['10123',2], ['10054',5], ['10456',3], ['10456',3], ['10456',3]]
        result = ['10444', 3, '10445', 4, '10456', 14, '10328', 5, '12266', 4, '10123', 2, '10054', 5]
        self.assertEqual(clean_credits_list(mess), result)
    
    def test_clean_credits_list_4(self):
        mess = [['40444',6],['40444',3],['40444',5]]
        self.assertEqual(clean_credits_list(mess), ['40444',14])
    
    def test_clean_credits_list_5(self):
        mess = []
        result = []
        self.assertEqual(clean_credits_list(mess), result)
    
    def test_clean_credits_list_6(self):
        mess = [['10456',5],['10456',5]]
        self.assertEqual(clean_credits_list(mess), ['10456',10])
    
    def test_clean_credits_list_7(self):
        mess = [['12999',3]]
        self.assertEqual(clean_credits_list(mess), ['12999',3])
    
    def test_clean_credits_list_8(self):
        mess = [['12999',3],['12999',3],['12999',3],['10456',5],['10456',5]]
        self.assertEqual(clean_credits_list(mess), ['12999',9,'10456',10])
    
    def test_clean_credits_list_9(self):
        mess = [['10444',3],['10445',4],['10456',5],['10328',5],['12266',4],['12266',4]]
        result = ['10444', 3, '10445', 4, '10456', 5, '10328', 5, '12266', 8]
        self.assertEqual(clean_credits_list(mess), result)

    ################################################################

# This class digs through AllTests, counts and builds all the tests,
# so that we have an entire test suite that can be run as a group.
class TheTestSuite (unittest.TestSuite):
    # constructor.
    def __init__(self,wants):
        self.num_req = 0
        self.num_ec = 0
        # find all methods that begin with "test".
        fs = []
        for w in wants:
            for func in AllTests.__dict__:
                # append regular tests
                # drop any digits from the end of str(func).
                dropnum = str(func)
                while dropnum[-1] in "1234567890":
                    dropnum = dropnum[:-1]
                
                if dropnum==("test_"+w+"_") and (not (dropnum==("test_extra_credit_"+w+"_"))):
                    fs.append(AllTests(str(func)))
                if dropnum==("test_extra_credit_"+w+"_") and not BATCH_MODE:
                    fs.append(AllTests(str(func)))
        
#       print("TTS ====> ",list(map(lambda f: (f,id(f)),fs)))
        # call parent class's constructor.
        unittest.TestSuite.__init__(self,fs)

class TheExtraCreditTestSuite (unittest.TestSuite):
        # constructor.
        def __init__(self,wants):
            # find all methods that begin with "test_extra_credit_".
            fs = []
            for w in wants:
                for func in AllTests.__dict__:
                    if str(func).startswith("test_extra_credit_"+w):
                        fs.append(AllTests(str(func)))
        
#           print("TTS ====> ",list(map(lambda f: (f,id(f)),fs)))
            # call parent class's constructor.
            unittest.TestSuite.__init__(self,fs)

# all (non-directory) file names, regardless of folder depth,
# under the given directory 'dir'.
def files_list(dir):
    this_file = __file__
    if dir==".":
        dir = os.getcwd()
    info = os.walk(dir)
    filenames = []
    for (dirpath,dirnames,filez) in info:
#       print(dirpath,dirnames,filez)
        if dirpath==".":
            continue
        for file in filez:
            if file==this_file:
                continue
            filenames.append(os.path.join(dirpath,file))
#       print(dirpath,dirnames,filez,"\n")
    return filenames

def main():
    if len(sys.argv)<2:
        raise Exception("needed student's file name as command-line argument:"\
            +"\n\t\"python3 testerX.py gmason76_2xx_Px.py\"")
    
    if BATCH_MODE:
        print("BATCH MODE.\n")
        run_all()
        return
        
    else:
        want_all = len(sys.argv) <=2
        wants = []
        # remove batch_mode signifiers from want-candidates.
        want_candidates = sys.argv[2:]
        for i in range(len(want_candidates)-1,-1,-1):
            if want_candidates[i] in ['.'] or os.path.isdir(want_candidates[i]):
                del want_candidates[i]
    
        # set wants and extra_credits to either be the lists of things they want, or all of them when unspecified.
        wants = []
        extra_credits = []
        if not want_all:
            for w in want_candidates:
                if w in REQUIRED_DEFNS:
                    wants.append(w)
                elif w in SUB_DEFNS:
                    wants.append(w)
                elif w in EXTRA_CREDIT_DEFNS:
                    extra_credits.append(w)
                else:
                    raise Exception("asked to limit testing to unknown function '%s'."%w)
        else:
            wants = REQUIRED_DEFNS + SUB_DEFNS
            extra_credits = EXTRA_CREDIT_DEFNS
        
        # now that we have parsed the function names to test, run this one file.    
        run_one(wants,extra_credits)    
        return
    return # should be unreachable! 

# only used for non-batch mode, since it does the printing.
# it nicely prints less info when no extra credit was attempted.
def run_one(wants, extra_credits):
    
    has_reqs = len(wants)>0
    has_ec   = len(extra_credits)>0
    
    # make sure they exist.
    passed1 = 0
    passed2 = 0
    tried1 = 0
    tried2 = 0
    
    # only run tests if needed.
    if has_reqs:
        print("\nRunning required definitions:")
        (tag, passed1,tried1) = run_file(sys.argv[1],wants,False)
    if has_ec:
        print("\nRunning extra credit definitions:")
        (tag, passed2,tried2) = run_file(sys.argv[1],extra_credits,True)
    
    # print output based on what we ran.
    if has_reqs and not has_ec:
        print("\n%d/%d Required test cases passed" % (passed1,tried1) )
      
    elif has_ec and not has_reqs:
        print("%d/%d Extra credit test cases passed (worth %d each)" % (passed2, tried2, weight_extra_credit))
    else: # has both, we assume.
        print("\n%d / %d Required test cases passed (worth %d each)" % (passed1,tried1,weight_required) )
        print("%d / %d Extra credit test cases passed (worth %d each)" % (passed2, tried2, weight_extra_credit))
        print("\nScore based on test cases: %.2f / %d ( %d * %d + %d * %d) " % (
                                                                passed1*weight_required+passed2*weight_extra_credit, 
                                                                total_points_from_tests,
                                                                passed1,
                                                                weight_required,
                                                                passed2,
                                                                weight_extra_credit
                                                             ))
    if CURRENTLY_GRADING:
        print("( %d %d %d %d )\n%s" % (passed1,tried1,passed2,tried2,tag))

# only used for batch mode.
def run_all():
        filenames = files_list(sys.argv[1])
        #print(filenames)
        
        wants = REQUIRED_DEFNS + SUB_DEFNS
        extra_credits = EXTRA_CREDIT_DEFNS
        
        results = []
        for filename in filenames:
            print(" Batching on : " +filename)
            # I'd like to use subprocess here, but I can't get it to give me the output when there's an error code returned... TODO for sure.
            lines = os.popen("python3 tester1p.py \""+filename+"\"").readlines()
            
            # delay of shame...
            time.sleep(DELAY_OF_SHAME)
            
            name = os.path.basename(lines[-1])
            stuff =lines[-2].split(" ")[1:-1]
            print("STUFF: ",stuff, "LINES: ", lines)
            (passed_req, tried_req, passed_ec, tried_ec) = stuff
            results.append((lines[-1],int(passed_req), int(tried_req), int(passed_ec), int(tried_ec)))
            continue
        
        print("\n\n\nGRAND RESULTS:\n")
        
            
        for (tag_req, passed_req, tried_req, passed_ec, tried_ec) in results:
            name = os.path.basename(tag_req).strip()
            earned   = passed_req*weight_required + passed_ec*weight_extra_credit
            possible = tried_req *weight_required # + tried_ec *weight_extra_credit
            print("%10s : %3d / %3d = %5.2d %% (%d/%d*%d + %d/%d*%d)" % (
                                                            name,
                                                            earned,
                                                            possible, 
                                                            (earned/possible)*100,
                                                            passed_req,tried_req,weight_required,
                                                            passed_ec,tried_ec,weight_extra_credit
                                                          ))
# only used for batch mode.
def run_all_orig():
        filenames = files_list(sys.argv[1])
        #print(filenames)
        
        wants = REQUIRED_DEFNS + SUB_DEFNS
        extra_credits = EXTRA_CREDIT_DEFNS
        
        results = []
        for filename in filenames:
            # wipe out all definitions between users.
            for fn in REQUIRED_DEFNS+EXTRA_CREDIT_DEFNS :
                globals()[fn] = decoy(fn)
                fn = decoy(fn)
            try:
                name = os.path.basename(filename)
                print("\n\n\nRUNNING: "+name)
                (tag_req, passed_req, tried_req) = run_file(filename,wants,False)
                (tag_ec,  passed_ec,  tried_ec ) = run_file(filename,extra_credits,True)
                results.append((tag_req,passed_req,tried_req,tag_ec,passed_ec,tried_ec))
                print(" ###### ", results)
            except SyntaxError as e:
                tag = filename+"_SYNTAX_ERROR"
                results.append((tag,0,len(wants),tag,0,len(extra_credits)))
            except NameError as e:
                tag =filename+"_Name_ERROR"
                results.append((tag,0,len(wants),tag,0,len(extra_credits)))
            except ValueError as e:
                tag = filename+"_VALUE_ERROR"
                results.append((tag,0,len(wants),tag,0,len(extra_credits)))
            except TypeError as e:
                tag = filename+"_TYPE_ERROR"
                results.append((tag,0,len(wants),tag,0,len(extra_credits)))
            except ImportError as e:
                tag = filename+"_IMPORT_ERROR_TRY_AGAIN"
                results.append((tag,0,len(wants),tag,0,len(extra_credits)))
            except Exception as e:
                tag = filename+str(e.__reduce__()[0])
                results.append((tag,0,len(wants),tag,0,len(extra_credits)))
        
#           try:
#               print("\n |||||||||| scrupe: "+str(scruples))
#           except Exception as e:
#               print("NO SCRUPE.",e)
#           scruples = None
        
        print("\n\n\nGRAND RESULTS:\n")
        for (tag_req, passed_req, tried_req, tag_ec, passed_ec, tried_ec) in results:
            name = os.path.basename(tag_req)
            earned   = passed_req*weight_required + passed_ec*weight_extra_credit
            possible = tried_req *weight_required # + tried_ec *weight_extra_credit
            print("%10s : %3d / %3d = %5.2d %% (%d/%d*%d + %d/%d*%d)" % (
                                                            name,
                                                            earned,
                                                            possible, 
                                                            (earned/possible)*100,
                                                            passed_req,tried_req,weight_required,
                                                            passed_ec,tried_ec,weight_extra_credit
                                                          ))

def try_copy(filename1, filename2, numTries):
    have_copy = False
    i = 0
    while (not have_copy) and (i < numTries):
        try:
            # move the student's code to a valid file.
            shutil.copy(filename1,filename2)
            
            # wait for file I/O to catch up...
            if(not wait_for_access(filename2, numTries)):
                return False
                
            have_copy = True
        except PermissionError:
            print("Trying to copy "+filename1+", may be locked...")
            i += 1
            time.sleep(1)
        except BaseException as e:
            print("\n\n\n\n\n\ntry-copy saw: "+e)
    
    if(i == numTries):
        return False
    return True

def try_remove(filename, numTries):
    removed = False
    i = 0
    while os.path.exists(filename) and (not removed) and (i < numTries):
        try:
            os.remove(filename)
            removed = True
        except OSError:
            print("Trying to remove "+filename+", may be locked...")
            i += 1
            time.sleep(1)
    if(i == numTries):
        return False
    return True

def wait_for_access(filename, numTries):
    i = 0
    while (not os.path.exists(filename) or not os.access(filename, os.R_OK)) and i < numTries:
        print("Waiting for access to "+filename+", may be locked...")
        time.sleep(1)
        i += 1
    if(i == numTries):
        return False
    return True

# this will group all the tests together, prepare them as 
# a test suite, and run them.
def run_file(filename,wants=None,checking_ec = False):
    if wants==None:
        wants = []
    
    # move the student's code to a valid file.
    if(not try_copy(filename,"student.py", 5)):
        print("Failed to copy " + filename + " to student.py.")
        quit()
        
    # import student's code, and *only* copy over the expected functions
    # for later use.
    import importlib
    count = 0
    while True:
        try:
#           print("\n\n\nbegin attempt:")
            while True:
                try:
                    f = open("student.py","a")
                    f.close()
                    break
                except:
                    pass
#           print ("\n\nSUCCESS!")
                
            import student
            importlib.reload(student)
            break
        except ImportError as e:
            print("import error getting student... trying again. "+os.getcwd(), os.path.exists("student.py"),e)
            time.sleep(0.5)
            while not os.path.exists("student.py"):
                time.sleep(0.5)
            count+=1
            if count>3:
                raise ImportError("too many attempts at importing!")
        except SyntaxError as e:
            print("SyntaxError in "+filename+":\n"+str(e))
            print("Run your file without the tester to see the details")
            return(filename+"_SYNTAX_ERROR",None, None, None)
        except NameError as e:
            print("NameError in "+filename+":\n"+str(e))
            print("Run your file without the tester to see the details")
            return((filename+"_Name_ERROR",0,1))    
        except ValueError as e:
            print("ValueError in "+filename+":\n"+str(e))
            print("Run your file without the tester to see the details")
            return(filename+"_VALUE_ERROR",0,1)
        except TypeError as e:
            print("TypeError in "+filename+":\n"+str(e))
            print("Run your file without the tester to see the details")
            return(filename+"_TYPE_ERROR",0,1)
        except ImportError as e:            
            print("ImportError in "+filename+":\n"+str(e))
            print("Run your file without the tester to see the details or try again")
            return((filename+"_IMPORT_ERROR_TRY_AGAIN   ",0,1)) 
        except Exception as e:
            print("Exception in loading"+filename+":\n"+str(e))
            print("Run your file without the tester to see the details")
            return(filename+str(e.__reduce__()[0]),0,1)
    
    # make a global for each expected definition.
    for fn in REQUIRED_DEFNS+EXTRA_CREDIT_DEFNS :
        globals()[fn] = decoy(fn)
        try:
            globals()[fn] = getattr(student,fn)
        except:
            if fn in wants:
                print("\nNO DEFINITION FOR '%s'." % fn) 
    
    if not checking_ec:
        # create an object that can run tests.
        runner = unittest.TextTestRunner()
    
        # define the suite of tests that should be run.
        suite = TheTestSuite(wants)
    
    
        # let the runner run the suite of tests.
        ans = runner.run(suite)
        num_errors   = len(ans.__dict__['errors'])
        num_failures = len(ans.__dict__['failures'])
        num_tests    = ans.__dict__['testsRun']
        num_passed   = num_tests - num_errors - num_failures
        # print(ans)
    
    else:
        # do the same for the extra credit.
        runner = unittest.TextTestRunner()
        suite = TheExtraCreditTestSuite(wants)
        ans = runner.run(suite)
        num_errors   = len(ans.__dict__['errors'])
        num_failures = len(ans.__dict__['failures'])
        num_tests    = ans.__dict__['testsRun']
        num_passed   = num_tests - num_errors - num_failures
        #print(ans)
    
    # remove our temporary file.
    os.remove("student.py")
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")
    if(not try_remove("student.py", 5)):
        print("Failed to remove " + filename + " to student.py.")
    
    tag = ".".join(filename.split(".")[:-1])
    
    
    return (tag, num_passed, num_tests)


# make a global for each expected definition.
def decoy(name):
        # this can accept any kind/amount of args, and will print a helpful message.
        def failyfail(*args, **kwargs):
            return ("<no '%s' definition was found - missing, or typo perhaps?>" % name)
        return failyfail

# this determines if we were imported (not __main__) or not;
# when we are the one file being run, perform the tests! :)
if __name__ == "__main__":
    main()

#####################################################################################
#   ****NOTE TO STUDENTS (READ)****:
#   YOU DO NOT need to read the code in this file. YOU also DO NOT need to read tests.txt
#   Ignore the weird formatting of output in tests.txt and DO NOT alter tests.txt
#   just make sure that tests.txt and tester1.py is in the SAME FOLDER as your CODE
#   and when you are ready to test your code, TYPE on your CMD or Terminal the following:
# 
#     Mac:        python3    tester1.py     yourfilehere.py
#     Windows:    python      tester1.py     yourfilehere.py
# 
# If you ever notice files named "student.py", "complaints.txt", or a folder 
# named "__pychache__", you can delete them. 
#  
# After testing your code, the only file you need to upload to Gradescope is your CODE 
# file named: netID_LabSection_Assignment.py
# Example: 
#         gmason76_230_PA1.py
#####################################################################################

import sys
import os
import io
import shutil
import time
import types #changed from imp to remove deprecation warning
import re

import traceback


# how many #'s to draw as a visual breaker?
banner_width = 30

# try to ignore everything we can.
def sanitize(message):
	# lower-case everything.
	message = message.lower()
	
	# normalize whitespace. Students might struggle here if
	# they *didn't* include any!
	message = "".join(message.split())
	# return what we've got.
	return message
	
def findStart(msg, i):
	v1 = msg.rfind('\n',0,max(0,i-5))
	v2 = msg.rfind('?',0,max(0,i-7))
	
	if v1 == -1 and v2 == -1:
		return 0
	elif v1 == -1:
		return v2+2
	elif v2 == -1:
		return v1+1
	else:
		return max(v1+1,v2+2)

def findEnd(msg, i):
	v1 = msg.find('\n',i+10)
	v2 = msg.find('?',i+10)
	
	if v1 == -1 and v2 == -1:
		return min(len(msg),i+30)
	elif v1 == -1:
		return v2+1
	elif v2 == -1:
		return v1-1
	else:
		return max(i+10,min(v1-1,v2+1))

def comp(expected, got):
	for i in range(max(len(expected),len(got))):
		if i >= len(expected) or i >= len(got) or expected[i] != got[i]:
			return "\tExpected:\n<<" + \
				repr(expected[findStart(expected, i-1):findEnd(expected, i+1)]) + \
				">>\n\n\tGot:\n<<" + \
				repr(got[findStart(got, i-1):findEnd(got, i+1)]) + \
				'>>'
	return ""

# put numbers on their own lines
def numbersOnSingleLine(message):
	length1 = 0
	
	#SPECIFIC TO THIS ASSIGNMENT
	message = re.sub("", "", message)
	
	message = re.sub("[^0-9]", " ", message)
	
	length2 = len(message)
	
	while(length1 != length2):
		message = re.sub("  ", " ", message)
		length1 = length2
		length2 = len(message)
		
	message = message.strip()
	
	return message
	
# builds up one single test case.
class Test:
	def __init__(self, inputs, expected, complaintsfilename = "complaints.txt"):
		self.inputs = inputs
		self.expected = expected
		#self.complaintsfile = open(complaintsfile,"a")
		self.complaintsfilename = complaintsfilename
	
	# group together list of inputs from many lines into one string.
	def prep_inputs(self,xs):
		s = ""
		for x in xs:
			s += str(x)+"\n"
		return s
	
	# maybe not needed anymore; way to reset stdin/out.
	# They get reassigned below.
	def reset(self):
		sys.stdin = sys.__stdin__
		sys.stdout = sys.__stdout__
	
		
	# run an individual test case. use other files
	# for inputs/outputs/error messages.	
	def run(self, outfilename = "results.txt", lenient=False):
		wait_for_access(self.complaintsfilename, 2, "Waiting for access to "+self.complaintsfilename+", may be locked...")
		
		self.complaintsfile = open(self.complaintsfilename,"a")
		answer = True
		try:
			# change stdin to a string(!)
			sys.stdin = io.StringIO(self.prep_inputs(self.inputs))
			
			sys.stdout = open(outfilename,"w")
			import student
			student.student_main()
			sys.stdout.close()
			
			wait_for_access(outfilename, 2, "Waiting for access to "+outfilename+", may be locked...")
			f = open(outfilename)
			got = f.read()
			f.close()
			
			if lenient:
				got = sanitize(got)
				self.expected = sanitize(self.expected)
			if got!=self.expected:
				hard_print("E",end="")
				
				numComp = ""
				# Uncomment for number comparison
				#gotNum = numbersOnSingleLine(got)
				#expectedNum = numbersOnSingleLine(self.expected)
				#if gotNum != expectedNum:
				#	numComp = "\n\n\tCalculations comparison:\n\n\tExpected:\n" + \
				#			  expectedNum + "\n\n\tGot:\n" + gotNum + "\n"
				
				#got = re.sub(" ", "_", got)
				#self.expected = re.sub(" ", "_", self.expected)
				
				print(("#"*banner_width)+"\n"
				      +"failure:\n"
					  +"\tfed these arguments:\n\t"+str(" ".join(self.inputs))
					  +"\n\n\tShowing area of differing output below:\n\n"
				      #+"    expected:\n"
					  #+'"""'+(repr(self.expected))
					  #+'"""\n    got:\n'
					  #+'"""'+(repr(got))+'"""\n'
					  +(comp(self.expected, got))+'\n'
					  + numComp
					  ,file=self.complaintsfile
					  )
				answer = False
			else:
				hard_print(".",end="")
				answer = True
		# sometimes the files are read/written/closed too soon or late. 
		# rather than deal with these race conditions (meaning waiting for
		# the files to be read/written/closed), we just ask for 
		# a re-run in these few cases. Sorry.
		except ValueError as ve:
			if str(ve).startswith("I/O operation on closed file."):
				hard_print("OOPS! poor timing... please run tests file "
							+"again! (closed file issue)")
				sys.exit(1)
			else:
				print(("#"*banner_width)+"\n"
					 +"failure: ",type(ve),ve
					 , file=self.complaintsfile)
				traceback.print_exception(*sys.exc_info(),file=self.complaintsfile)
			answer = False
		except ImportError as ie:
			if str(ie).startswith("No module named 'student'"):
				hard_print("OOPS! poor timing... please run tests file "
							+"again! (imported before file was created)")
				sys.exit(1)
			answer = False
		except FileNotFoundError as e:
			if str(e).startswith("[Errno 2] No such file or directory:"
							+" 'complaints.txt'"):
				hard_print("OOPS! poor timing... please run tests file "
							+"again! (complaints.txt wasn't yet created)")
				sys.exit(1)
			answer = False
		# anything else that goes wrong is supposedly an
		# actual student-code failure.
		except Exception as e:
			hard_print("F",end="")
			print(("#"*banner_width)+"\n"
				 +"failure: ",type(e),e
				 , file=self.complaintsfile)
			traceback.print_exception(*sys.exc_info(),file=self.complaintsfile)
			answer = False
		#Warning if you use exit() or quit() or sys.exit()	
		except SystemExit as se:
			hard_print("W",end="")
			print(("#"*banner_width)+"\n"
				 +"warning: used exit()/quit()/sys.exit()"
				 , file=self.complaintsfile)
			answer = False
		
		self.reset()
		self.complaintsfile.close()
		return answer

# batch of Test values.
class Tests:
	def __init__(self):
		self.tests = []
	def add(self, test):
		self.tests.append(test)
	# run them all. can be done leniently (massaging string formatting/spacing
	# for better matching possibilities).
	def run(self, printout=True, lenient=False):
		# empty the complaints file.
		open("complaints.txt","w").close()
		
		count = 0
		for test in self.tests:
			if test.run(lenient=lenient):
				count+=1
		if printout:
			try:
				wait_for_access("complaints.txt", 2, "Waiting for access to complaints.txt, may be locked...");
				f = open("complaints.txt")
				s = f.read()
				if s!="":
					print("\n"+s)
				f.close()
				
				try_remove("complaints.txt", 2, "Trying to remove complaints.txt, may be locked...");
				if os.path.exists("complaints.txt"):
					os.remove("complaints.txt")
			except FileNotFoundError as e:
				if str(e).startswith("[Errno 2] No such file or directory: 'complaints.txt'"):
					hard_print("OOPS! poor timing... please run tests file again! (complaints.txt wasn't yet created)")
					sys.exit(1)
		return (count,len(self.tests))

# dig through the files' INPUT/OUTPUT/ENDTESTS style.
def build_tests(filename):
	wait_for_access(filename, 2, "Waiting for access to "+filename+", may be locked...")
	with open(filename) as f:
		lines = f.read().split("\n")
	tests = Tests()
	while len(lines)>0 and lines[0] =="INPUT":
		lines.pop(0) # drop "INPUT" line
		ins = []
		while len(lines)>0 and lines[0]!="OUTPUT":
			ins.append(lines.pop(0))
		if len(lines)>0:
			lines.pop(0) # drop "OUTPUT" line
		outs = []
		while len(lines)>0 and not (lines[0] in ["INPUT","ENDTESTS"]):
			outs.append(lines.pop(0))
		outstr = "\n".join(outs)
		#hard_print("outstr: [[["+outstr+"]]]")
		tests.add(Test(ins,outstr))
	return tests

# add depth spaces before each line in this assumedly multi-line string s.
def indent(s, depth=6):
	depth_str = " "*depth
	return "\n".join(map(lambda line : (depth_str+line),s.split("\n")))
	
def import_student_code(filename):
	prefix = "def student_main():\n"
	postfix = "\n"
	
	f = open(filename)
	code = prefix+indent(f.read())+postfix
	f.close()
	
	module = types.ModuleType("student")
	sys.modules["student"] = module
	exec(code, module.__dict__)
	
	return module

def wait_for_access(filename, numTries, message):
	i = 0
	while (not os.path.exists(filename) or not os.access(filename, os.R_OK)) and i < numTries:
		print(message)
		time.sleep(1)
		i += 1
	if(i == numTries):
		return False
	return True
			
def try_remove(filename, numTries, message):
	removed = False
	i = 0
	while os.path.exists(filename) and (not removed) and (i < numTries):
		try:
			os.remove(filename)
			removed = True
		except OSError:
			print(message)
			i += 1
			time.sleep(1)
	if(i == numTries):
		return False
	return True

# put all their code in another file, also forcing it to be a function
# named student_main (indenting all they do).
def try_copy(filename1, filename2, numTries):
	have_copy = False
	i = 0
	while (not have_copy) and (i < numTries):
		try:
			# move the student's code to a valid file.
			prefix = "def student_main():\n"
			postfix = "\n"
			
			f = open(filename1)
			code = f.read()
			f.close()
			
			f = open(filename2,"w")
			f.write(prefix+indent(code)+postfix)
			f.close()
			
			# wait for file I/O to catch up...
			if(not wait_for_access(filename2, numTries, "Waiting for access to "+filename2+", may be locked...")):
				return False
				
			have_copy = True
		except PermissionError:
			print("Trying to copy "+filename1+", may be locked...")
			i += 1
			time.sleep(1)
	
	if(i == numTries):
		return False
	return True
# not really needed anymore; just always wrote to stdout even during a program
# run that was recording outputs to a file with a repurposed print() target.
def hard_print(*args,**kwargs):
	print(*args, file=sys.__stdout__,**kwargs)

# relocate student's code; build all tests; run them; report answers.
def main():
	# empty student.py
	#open("student.py","w").close()
	#try_copy(sys.argv[1], "student.py", 2)
	import_student_code(sys.argv[1])
	all_good = True
	count = 0
	tests = build_tests("tests.txt")
	(passed,total) = tests.run()
	print("\n\n"+("+"*40)+"\n"+("+"*40)+"\n")
	print("\npassed %d/%d tests." % (passed,total))
	
	try_remove("student.py", 2, "Trying to remove student.py, may be locked...")
	try_remove("results.txt", 2, "Trying to remove results.txt, may be locked...")
	if os.path.exists("__pycache__"):
		shutil.rmtree("__pycache__")


# run it all.
if __name__ == "__main__":
	main()

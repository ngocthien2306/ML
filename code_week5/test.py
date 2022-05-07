# Python3 program for implementation of FCFS
# scheduling with different arrival time
 
# Function to find the waiting time
# for all processes
def findWaitingTime(processes, n, bt, wt, at):
    service_time = [0] * n
    service_time[0] = 0
    wt[0] = 0
 
    # calculating waiting time
    for i in range(1, n):
      # Add burst time of previous processes
      service_time[i] = (service_time[i - 1] + bt[i - 1])

      # Find waiting time for current
      # process = sum - at[i]
      wt[i] = service_time[i] - at[i]

      # If waiting time for a process is in
      # negative that means it is already
      # in the ready queue before CPU becomes
      # idle so its waiting time is 0
      if (wt[i] < 0):
          wt[i] = 0
     
# Function to calculate turn around time
def findTurnAroundTime(processes, n, bt, wt, tat):
     
    # Calculating turnaround time by
    # adding bt[i] + wt[i]
    for i in range(n):
        tat[i] = bt[i] + wt[i]
 
 
# Function to calculate average waiting
# and turn-around times.
def findavgTime(processes, n, bt, at):
    wt = [0] * n
    tat = [0] * n
 
    # Function to find waiting time
    # of all processes
    findWaitingTime(processes, n, bt, wt, at)
 
    # Function to find turn around time for
    # all processes
    findTurnAroundTime(processes, n, bt, wt, tat)
 
    # Display processes along with all details
    print("Processes   Burst Time   Arrival Time     Waiting",
          "Time   Turn-Around Time  Completion Time \n")
    total_wt = 0
    total_tat = 0
    for i in range(n):
 
        total_wt = total_wt + wt[i]
        total_tat = total_tat + tat[i]
        compl_time = tat[i] + at[i]
        print(" ", i + 1, "\t\t", bt[i], "\t\t", at[i],
              "\t\t", wt[i], "\t\t ", tat[i], "\t\t ", compl_time)
 
    print("Average waiting time = %.5f "%(total_wt /n))
    print("\nAverage turn around time = ", total_tat / n)
 
# Driver code
if __name__ =="__main__":
     
    # Process id's
    processes = [1, 2, 3, 4]
    n = 4
 
    # Burst time of all processes
    burst_time = [8, 4, 9, 5]
 
    # Arrival time of all processes
    arrival_time = [0, 1, 2, 3]
 
    findavgTime(processes, n, burst_time, arrival_time)
 
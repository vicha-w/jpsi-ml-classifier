import os
import argparse
import json
import math

parser = argparse.ArgumentParser(description='Prepare training file')
parser.add_argument('file_name', metavar='filename', type=str, nargs=1,
                   help='Name of input file to be used')
parser.add_argument('out_file', metavar='outfilename', type=str, help='Name of output file', default='Train.txt')
args = parser.parse_args()
file_name = args.file_name[0]
out_file_name = args.out_file

if not os.path.isfile(file_name):
    print("File not available.")
    print("Try again, baby.")
    exit()

targetFileName = open(args.file_name[0])
events = targetFileName.readlines()[2:]

outfile = open(out_file_name,'w')

for line in events:
    eventNum = int(line.split()[2])
    gradeNum = int(line.split()[3])
    eventFileName = 'Run_140124/Event_'+str(eventNum)
    if not os.path.isfile(eventFileName):
        print("WARNING: Event {} not available in events folder. Skipping.".format(eventNum))
        continue

    eventstr = ''
    eventstr += str(eventNum) + '\t'
    eventstr += str(gradeNum) + '\t'

    eventFile = open(eventFileName)
    allLines = ' '.join(eventFile.readlines())
    allLines = allLines.replace('\'', '"')
    allLines = allLines.replace('(', '[')
    allLines = allLines.replace(')', ']')
    allLines = allLines.replace('nan','null')
    parsedEvent = json.loads(allLines)
    #print(parsedEvent['Collections']['TrackerMuons_V1'])
    trackerMuons = parsedEvent['Collections']['TrackerMuons_V1']
    standaloneMuons = parsedEvent['Collections']['StandaloneMuons_V2']
    globalMuons = parsedEvent['Collections']['GlobalMuons_V1']
    extrasPoints = parsedEvent['Collections']['Extras_V1']
    pointsPoints = parsedEvent['Collections']['Points_V1']

    muonTrackExtras = parsedEvent['Associations']['MuonTrackExtras_V1'] # for standaloneMuons
    muonTrackerPoints = parsedEvent['Associations']['MuonTrackerPoints_V1'] # for trackerMuons
    muonGlobalPoints = parsedEvent['Associations']['MuonGlobalPoints_V1'] # for globalMuons

    trackerNum = len(trackerMuons)
    standaloneNum = len(standaloneMuons)
    globalNum = len(globalMuons)

    #eventstr += str(trackerNum) + '\t'
    #eventstr += str(standaloneNum) + '\t' 
    #eventstr += str(globalNum) + '\t'
    eventstrDisp = ''

    fourMuons = []
    startPointMuons = []
    endPointMuons = []
    invCurvatureMuons = []
    invCurvatureMuons2 = []

    #print(line)

    for i in range(trackerNum):
        # eventstrDisp += str(trackerMuons[i][0]) + '\t' # pt
        # eventstrDisp += str(trackerMuons[i][1]) + '\t' # charge
        # eventstrDisp += str(trackerMuons[i][2][0]) + '\t' # rpx
        # eventstrDisp += str(trackerMuons[i][2][1]) + '\t' # rpy
        # eventstrDisp += str(trackerMuons[i][2][2]) + '\t' # rpz
        # eventstrDisp += str(trackerMuons[i][3]) + '\t' # phi
        # eventstrDisp += str(trackerMuons[i][4]) + '\t' # eta
        # eventstrDisp += str(trackerMuons[i][5]) + '\t' # calo_energy
        #eventstr += '\t'.join([str(num) for num in (event[3:7] if i==0 else event[11:15])]) # E and p
        if gradeNum == 0:
            fourMuons.append(line.split()[4:8] if i==0 else line.split()[8:12])
        else:
            fourMuons.append(line.split()[5:9] if i==0 else line.split()[9:13])

        charge = float(trackerMuons[i][1])

        fourMomenta = [float(num) for num in fourMuons[i]]
        #print(fourMomenta)
        mass = fourMomenta[0]**2 - fourMomenta[1]**2 - fourMomenta[2]**2 - fourMomenta[3]**2
        vt = math.sqrt((fourMomenta[1]/fourMomenta[0])**2 + (fourMomenta[2]/fourMomenta[0])**2)

        invCurvatureMuons.append(charge/mass/vt)
    
    trackerMuonObj = []
    globalMuonObj  = []

    for i in range(trackerNum): 
        cache = []
        for j in [0,1,4,3]:
            cache.append(float(trackerMuons[i][j])) #pt charge eta phi
        startPoint = -1
        endPoint = -1
        for pair in muonTrackerPoints:
            if pair[0][0] != 29 or pair[0][1] != i: continue
            if startPoint == -1: startPoint = int(pair[1][1])
            elif int(pair[1][1]) < startPoint: startPoint = int(pair[1][1])
            if endPoint == -1: endPoint = int(pair[1][1])
            elif int(pair[1][1]) > endPoint: endPoint = int(pair[1][1])
        for j in range(3):
            if startPoint < 0: cache.append(0)
            else: cache.append(pointsPoints[startPoint][0][j])
        for j in range(3):
            if endPoint < 0: cache.append(0)
            else: cache.append(pointsPoints[endPoint][0][j])
        trackerMuonObj.append(cache)

    for i in range(globalNum): 
        cache = []
        for j in [0,1,4,3]:
            cache.append(float(globalMuons[i][j])) #pt charge eta phi
        startPoint = -1
        endPoint = -1
        for pair in muonGlobalPoints:
            if pair[0][0] != 31 or pair[0][1] != i: continue
            if startPoint == -1: startPoint = int(pair[1][1])
            elif int(pair[1][1]) < startPoint: startPoint = int(pair[1][1])
            if endPoint == -1: endPoint = int(pair[1][1])
            elif int(pair[1][1]) > endPoint: endPoint = int(pair[1][1])
        for j in range(3):
            if startPoint < 0: cache.append(0)
            else: cache.append(pointsPoints[startPoint][0][j])
        for j in range(3):
            if endPoint < 0: cache.append(0)
            else: cache.append(pointsPoints[endPoint][0][j])
        globalMuonObj.append(cache)

    if globalNum == 2:
        deltaR = []
        deltaR.append(
            math.sqrt((globalMuonObj[0][2]-trackerMuonObj[0][2])**2 + (globalMuonObj[0][3]-trackerMuonObj[0][3])**2) +
            math.sqrt((globalMuonObj[1][2]-trackerMuonObj[1][2])**2 + (globalMuonObj[1][3]-trackerMuonObj[1][3])**2)
        )
        deltaR.append(
            math.sqrt((globalMuonObj[0][2]-trackerMuonObj[1][2])**2 + (globalMuonObj[0][3]-trackerMuonObj[1][3])**2) +
            math.sqrt((globalMuonObj[1][2]-trackerMuonObj[0][2])**2 + (globalMuonObj[1][3]-trackerMuonObj[0][3])**2)
        )
        if deltaR[0] < deltaR[1]:
            startPointMuons.append(globalMuonObj[0][4:7])
            startPointMuons.append(globalMuonObj[1][4:7])
            endPointMuons.append(globalMuonObj[0][7:])
            endPointMuons.append(globalMuonObj[1][7:])
        else:
            startPointMuons.append(globalMuonObj[1][4:7])
            startPointMuons.append(globalMuonObj[0][4:7])
            endPointMuons.append(globalMuonObj[1][7:])
            endPointMuons.append(globalMuonObj[0][7:])
        
        invCurvatureMuons2 = [-c/2 for c in invCurvatureMuons]
    elif globalNum == 1:
        deltaR = []
        deltaR.append(
            math.sqrt((globalMuonObj[0][2]-trackerMuonObj[0][2])**2 + (globalMuonObj[0][3]-trackerMuonObj[0][3])**2)
        )
        deltaR.append(
            math.sqrt((globalMuonObj[0][2]-trackerMuonObj[1][2])**2 + (globalMuonObj[0][3]-trackerMuonObj[1][3])**2)
        )
        if deltaR[0] < deltaR[1]:
            startPointMuons.append(globalMuonObj[0][4:7])
            startPointMuons.append(trackerMuonObj[1][4:7])
            endPointMuons.append(globalMuonObj[0][7:])
            endPointMuons.append(trackerMuonObj[1][7:])

            invCurvatureMuons2.append(-invCurvatureMuons[0]/2)
            invCurvatureMuons2.append(invCurvatureMuons[1])
        else:
            startPointMuons.append(trackerMuonObj[0][4:7])
            startPointMuons.append(globalMuonObj[0][4:7])
            endPointMuons.append(trackerMuonObj[0][7:])
            endPointMuons.append(globalMuonObj[0][7:])

            invCurvatureMuons2.append(invCurvatureMuons[0])
            invCurvatureMuons2.append(-invCurvatureMuons[1]/2)
    else:
        startPointMuons.append(trackerMuonObj[0][4:7])
        startPointMuons.append(trackerMuonObj[1][4:7])
        endPointMuons.append(trackerMuonObj[0][7:])
        endPointMuons.append(trackerMuonObj[1][7:])
        invCurvatureMuons2 = invCurvatureMuons[:]
    
    startPointMuonsCylind = []
    endPointMuonsCylind = []

    for muon in startPointMuons:
        rho = math.sqrt(muon[0]**2 + muon[1]**2)
        if muon[0] != 0: phi = math.atan(muon[1]/muon[0])
        else: phi = math.copysign(math.pi/2, muon[1])
        startPointMuonsCylind.append([rho, phi, muon[2]])
        
    for muon in endPointMuons:
        rho = math.sqrt(muon[0]**2 + muon[1]**2)
        if muon[0] != 0: phi = math.atan(muon[1]/muon[0])
        else: phi = math.copysign(math.pi/2, muon[1])
        endPointMuonsCylind.append([rho, phi, muon[2]])

    for i in [0, 1]:
        eventstr += '\t'.join([str(num) for num in fourMuons[i]]) + '\t'
        eventstr += '\t'.join([str(num) for num in startPointMuonsCylind[i]]) + '\t'
        eventstr += '\t'.join([str(num) for num in endPointMuonsCylind[i]]) + '\t'
        eventstr += str(invCurvatureMuons[i]) + '\t'
        eventstr += str(invCurvatureMuons2[i]) + '\t'
    
    #print(eventstr.replace('\t',' '))
    eventstr += '\n'
    outfile.write(eventstr)
    
outfile.close()
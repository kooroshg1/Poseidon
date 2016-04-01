function deleteFiles()
try
    delete('boundaryCondition.txt');
    delete('information.txt');
    delete('lagrangePointsVelocity.txt');
    delete('pointCloud.txt');
    delete('U.txt');
    delete('V.txt');
    delete('P.txt');
catch
end
function [ d ] = Distance(A, B)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

d = sqrt(sum((A-B).^2));

end


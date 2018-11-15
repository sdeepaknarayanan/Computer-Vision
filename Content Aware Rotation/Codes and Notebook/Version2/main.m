

filepath = '/Dataset/image9.jpg';
p = getParams(delta, filepath)
V = csvread('/Results/Vertices9.csv');
V = V(:,2);
V = V(2:size(V));
Vx = V(2*p.k-1);
Vy = V(2*p.k);
I = warpMesh(p,Vx,Vy);
imshow(I)
imwrite(I, '/Current Results/image9.jpg');



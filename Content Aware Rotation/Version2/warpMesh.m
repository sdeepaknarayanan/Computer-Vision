function I = warpMesh(p,Vx,Vy)

    I = zeros(size(p.I));
    lineX = 1:1:p.X;
    numX = length(lineX);
    lineY = (1:1:p.Y)';
    numY = length(lineY);
    gridX = ones(numY,1)*lineX;
    gridY = lineY*ones(1,numX);
    sampleX = gridX(:);
    sampleX = sampleX';
    sampleY = gridY(:);
    sampleY = sampleY';
    sampleXY = [sampleX;sampleY];
    ori_sampleXY = zeros(size(sampleXY));
    tempI = zeros(3,size(sampleXY,2));
    count = zeros(3,size(sampleXY,2));

    % Barycentric coordinates
	% We are trating each quad as two triangles
	% This is very crucial in this implementation as it helps
	% in ensuring that we can transform or warp each pixel to its corresponding
	% Warped location in the new image

    for i = 1 : p.cx
        for j = 1 : p.cy
            v4 = [(i-1)*(p.cy+1)+j,(i-1)*(p.cy+1)+j+1,i*(p.cy+1)+j,i*(p.cy+1)+j+1];	% Represents the 4 quad vertices
            for k = 1 : 2								% To cover both the triangles
                if k == 1
                   
                   triangle = [v4(1) v4(2) v4(4)];					% Choosing the vertices of the traingle

                elseif k == 2
            
                   triangle = [v4(1) v4(3) v4(4)]; 					% Choosing the vertices of the traingle
                end

                cornerX = Vx(triangle);							% Vx has all the warped coordinates - we're getting the X Coordinates of the triangle
                cornerY = Vy(triangle);							% Vy has all the warped coordinates - we're getting the Y Coordinates of the traingle

                cornerXY = [cornerX'; cornerY'];

		T = cornerXY(:,1:2)-cornerXY(:,3)*ones(1,2);				% Any point inside the traingle can be represented as a linear combination of the 
											% Coordinates of the three points forming the traingle with the coeeficients all,
											% all adding upto 1.

                d_sampleXY = sampleXY-cornerXY(:,3)*ones(1,size(sampleXY,2));
             
                lambda = inv(T)*d_sampleXY;						% We've found the lambda for all the points that we need - all dicrete points
                lambda = [lambda; 1-lambda(1,:)-lambda(2,:)];				% We have the relation that the sum of lambdas is 1 - that is why, this equation
                index_table = (lambda(1,:)<= 1).*(lambda(2,:)<=1).*(lambda(3,:)<=1);	% We check only where all the lambdas have value <=1 - This is a 
                idx = find(index_table == 1);
        
                original_cornerX = p.Vx(triangle);					
                original_cornerY = p.Vy(triangle);
                original_cornerXY = [original_cornerX'; original_cornerY'];

                temp_sampleXY = original_cornerXY*lambda;				% Gives us the tranformation of the new coordinate for our new quad.	

                tempI(1,idx) =  tempI(1,idx) + interp2(double(p.I(:,:,1)),temp_sampleXY(1,idx),temp_sampleXY(2,idx),'*linear');
                tempI(2,idx) =  tempI(2,idx) + interp2(double(p.I(:,:,2)),temp_sampleXY(1,idx),temp_sampleXY(2,idx),'*linear');
                tempI(3,idx) =  tempI(3,idx) + interp2(double(p.I(:,:,3)),temp_sampleXY(1,idx),temp_sampleXY(2,idx),'*linear');
                count(:,idx) = count(:,idx)+1;
            end
        end
    end

    tempI =  tempI./count;								% Interp2 in MATLAB is a very special function. I've used this because of the fact
											% that it supports the kind of interpolation that I need - Python doesn't have a "proper"
											% Equivalent of Interp2.

    k_idx = find(isnan(tempI(1,:))|isinf(tempI(1,:))|isnan(tempI(2,:))|isinf(tempI(2,:))|isnan(tempI(3,:))|isinf(tempI(3,:)));
    tempI(:,k_idx) = 0;
											% Checking for all possible cases that couldl occur.
    I(:,:,1) = reshape(tempI(1,:)',numY,numX);
    I(:,:,2) = reshape(tempI(2,:)',numY,numX);						% Fill up the RGB's!!
    I(:,:,3) = reshape(tempI(3,:)',numY,numX);
											
    I = I(2:end-1,2:end-1,:);										
    I = uint8(I);	


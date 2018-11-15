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
    for i = 1 : p.cx
        for j = 1 : p.cy
            v4 = [(i-1)*(p.cy+1)+j,(i-1)*(p.cy+1)+j+1,i*(p.cy+1)+j,i*(p.cy+1)+j+1];
            for k = 1 : 2
                if k == 1
                   % first triangle---v1 v2 v4
                   tri_v = [v4(1) v4(2) v4(4)];
                elseif k == 2
                   % first triangle---v1 v3 v4
                   tri_v = [v4(1) v4(3) v4(4)]; 
                end
                cornerX = Vx(tri_v);
                cornerY = Vy(tri_v);
                cornerXY = [cornerX'; cornerY'];

                d_sampleXY = sampleXY-cornerXY(:,3)*ones(1,size(sampleXY,2));
                T = cornerXY(:,1:2)-cornerXY(:,3)*ones(1,2);
                lambda = inv(T)*d_sampleXY;
                lambda = [lambda; 1-lambda(1,:)-lambda(2,:)];
                index_table = (lambda(1,:)<= 1).*(lambda(2,:)<=1).*(lambda(3,:)<=1);
                idx = find(index_table == 1);
        
                ori_cornerX = p.Vx(tri_v);
                ori_cornerY = p.Vy(tri_v);
                ori_cornerXY = [ori_cornerX'; ori_cornerY'];
                temp_sampleXY = ori_cornerXY*lambda;
                tempI(1,idx) =  tempI(1,idx) + interp2(double(p.I(:,:,1)),temp_sampleXY(1,idx),temp_sampleXY(2,idx),'*linear');
                tempI(2,idx) =  tempI(2,idx) + interp2(double(p.I(:,:,2)),temp_sampleXY(1,idx),temp_sampleXY(2,idx),'*linear');
                tempI(3,idx) =  tempI(3,idx) + interp2(double(p.I(:,:,3)),temp_sampleXY(1,idx),temp_sampleXY(2,idx),'*linear');
                count(:,idx) = count(:,idx)+1;
            end
        end
    end
    tempI =  tempI./count;
    k_idx = find(isnan(tempI(1,:))|isinf(tempI(1,:))|isnan(tempI(2,:))|isinf(tempI(2,:))|isnan(tempI(3,:))|isinf(tempI(3,:)));
    tempI(:,k_idx) = 0;
    I(:,:,1) = reshape(tempI(1,:)',numY,numX);
    I(:,:,2) = reshape(tempI(2,:)',numY,numX);
    I(:,:,3) = reshape(tempI(3,:)',numY,numX);
    I = I(2:end-1,2:end-1,:);
    I = uint8(I);
%    imshow(I);
%     image(uint8(I));
%     imwrite(uint8(I),'rotate_fig2.png');

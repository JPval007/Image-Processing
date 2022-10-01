% =========================================================================
% Image processing with OpenCV
% =========================================================================

%Import images to process
%Cameraman
Camera = imread('standard_test_images/cameraman.tif');
%House
House = imread('standard_test_images/house.tif');
House = House(:,:,1); %por alguna razón es un array 3D
%For some reason this one can't be shown
%Livingroom
Liv = imread('standard_test_images/livingroom.tif');

% imshow(Camera)
%  imshow(House) tiene algo raro, un truco
% imshow(Liv)

%%Camera man
%Histogram visualization
% % figure;
% % imhist(Camera)
% % title('Histogram visualization')
% % ylabel('No. pixels')
% % xlabel('Bins')
%Histogram equalization
% % figure;
% % Camera_eq = histeq(Camera);
% % imhist(Camera_eq)
% % title('Histogram equalization')
% % ylabel('No. pixels')
% % xlabel('Bins')
%Gaussian blur 5x5
% % Gaussb = imgaussfilt(Camera,3);
% % figure;
% % subplot(1,2,1);
% % imshow(Camera)
% % title('Original')
% % subplot(1,2,2);
% % imshow(Gaussb)
% % title('Gaussian blur 5x5')


%%House
%Sobel edge detection
% BW1 = edge(House(:,:,1),'sobel');
% imshow(BW1)
% title('Sobel edge detection')

%Harris corner detection
I = House;
%corners = detectFASTFeatures(I);
corners = detectHarrisFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(50));

%Canny edge detection
% BW1 = edge(House,'canny');
% imshow(BW1)


%Hough transform
%rotI = imrotate(House,33,'crop');
% % rotI = House;
% % BW = edge(rotI,'canny');
% % [H,T,R] = hough(BW);
% % imshow(H,[],'XData',T,'YData',R,...
% %             'InitialMagnification','fit');
% % xlabel('\theta'), ylabel('\rho');
% % axis on, axis normal, hold on;
% % 
% % P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
% % x = T(P(:,2)); y = R(P(:,1));
% % plot(x,y,'s','color','white');
% % 
% % lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',7);
% % figure, imshow(rotI), hold on
% % max_len = 0;
% % for k = 1:length(lines)
% %    xy = [lines(k).point1; lines(k).point2];
% %    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% % 
% %    % Plot beginnings and ends of lines
% %    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
% %    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% % 
% %    % Determine the endpoints of the longest line segment
% %    len = norm(lines(k).point1 - lines(k).point2);
% %    if ( len > max_len)
% %       max_len = len;
% %       xy_long = xy;
% %    end
% % end

%plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');

%Livingroom
% points = detectSIFTFeatures(Liv);
% imshow(Liv);
% hold on;
% plot(points.selectStrongest(10))





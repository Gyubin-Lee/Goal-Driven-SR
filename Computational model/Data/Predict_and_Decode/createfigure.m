function createfigure(X1, Y1, D1)
%CREATEFIGURE(X1, Y1, D1)
%  X1:  scatter x
%  Y1:  scatter y
%  D1:  errorbar delta 벡터 데이터

%  MATLAB에서 15-Feb-2022 17:09:39에 자동 생성됨

% figure 생성
figure('OuterPosition',[2380 392 576 514]);

% axes 생성
axes1 = axes;
hold(axes1,'on');

% scatter 생성
scatter(X1,Y1,'MarkerEdgeColor',[0 0 0],'Marker','o');

% errorbar 생성
errorbar(X1,Y1,D1,'horizontal','LineStyle','none','Color',[0 0 0]);

% ylabel 생성
ylabel('Actual CES-D');

% xlabel 생성
xlabel('Predicted CES-D');

% 다음 라인의 주석을 해제하여 좌표축의 X 제한을 유지
xlim(axes1,[7 23]);
box(axes1,'on');
hold(axes1,'off');
% 나머지 axes 속성 설정
set(axes1,'FontSize',20,'LineWidth',2);

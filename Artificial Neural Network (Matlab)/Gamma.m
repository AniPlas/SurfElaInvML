%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This MATLAB code computes the surface excess energy density (Gamma) as  %
% a function of surface elastic invariants and in-plane deformation for 7 %
% face-centered cubic (FCC) metals (Ag, Al, Au, Cu, Ni, Pd and Pt).       %
% The surface invariants represent the surface elastic properties,        %
% including the residual surface stress tensor (Sigma) and the surface    %
% elastic stiffness tensor (C), within a polar method.                    %
%                                                                         %
% Only the surface orientation (represented by two angles theta and phi)  %
% is considered as the input for the developed ANN models.                %
%                                                                         %
% In this calculation, all the surface elastic invariants are predicted   %
% by the developed artificial neural network (ANN) models. The trained    %
% ANN models are stored in the file "Networks_FCC_FreeSurface.mat".       %
% Users can train their own ANN models with different ANN structures,     %
% using the provided code "TrainNetworks.m" based on the results from the %
% semi-analytical calculation.                                            %
%                                                                         %
% In the code, the output variables "Strain_Magnitude", "Gamma_Total",    %
% "Gamma_Stress" and "Gamma_Stiffness" indicate the magnitude of the      %
% applied in-plane strain, the total of Gamma, the contribution of Sigma  %
% for Gamma and the contribution of C for Gamma, respectively.            %
% Meanwhile, a Figure of the evaluation of Gamma as a function of the     %
% applied in-plane strain magnitude is created, with the name of the form %
% "Gamma_Strain Mode_on_Material_Surface orientation_with reference_      %
% Reference Principal Direction=Deflection Angle_degree and nu=           %
% Concentration Factor_Contribution or Not".                              %                       
%                                                                         %
% For more details, please check in the reference paper.                  %
%                                                                         %
% References                                                              %
%                                                                         %
% [1] X. Chen, R. Dingreville, T. Richeton and S. Berbenni.               %
%                                                                         %
% Invariant surface elastic properties in FCC metals via machine learning %
% methods. Submission to Journal of the Mechanics and Physics of Solids.  %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Main program
clear all;
close all;
% Indicate the studied material in the list:
% ['Ag','Al','Au','Cu','Ni','Pd','Pt'].
Material='Cu';
% Indicate the normal vector of the studied surface.
n_surface=[1,0,0];
% Indicate the concentration factor nu for angle predictions.
nu=0.001;
% Indicate the deformation mode in the list: ['Uniaxial_Compression','Uniaxial_Tension',
% 'Biaxial_Compression','Biaxial_Tension','Positive_Shear','Negative_Shear'].
ModeDeform='Uniaxial_Compression';
% Indicate the reference principal direction in the list: ['Phi','Phi1','Phi0'].
Ref_Phi='Phi';
% Indicate the angle between the strain principal direction and the reference direction. 
Angle=0; % in degree
% Indicate the maximum applied strain
epsilon_max=0.2;
% If present the contribution of the residual stress and the stiffness, Contribution='Yes', 
% if not, Contribution='No'.
Contribution='Yes';
% Use multiple CPU cores on a single PC, or across multiple CPUs on multiple computers on a network using MATLAB Parallel Server. If not, Parallel='No'.
Parallel='Yes'; 

[Strain_Magnitude,Gamma_Total,Gamma_Stress,Gamma_Stiffness]=PlotGamma(Material,n_surface,nu,ModeDeform,Ref_Phi,Angle,epsilon_max,Contribution,Parallel);

%% Main function
function [Strain_Magnitude,Gamma_Total,Gamma_Stress,Gamma_Stiffness]=PlotGamma(Material,n_surface,nu,ModeDeform,Ref_Phi,Angle,epsilon_max,Contribution,Parallel)
    %% Define the input factors for neural networks
    load('Networks_FCC_FreeSurface.mat');
    n_surface=SIPF(n_surface,T_sym);
    InputPara=zeros(2,1);
    InputPara(1,1)=acos(n_surface(:,3))/pi*180;
    InputPara(2,1)=atan(n_surface(:,2)./n_surface(:,1))/pi*180;

    %% Regression for Phi
    VarPhiList=["Phi1_Phi","Phi0_Phi1"];
    N_Var_Phi=length(VarPhiList);
    VarRange=[180,90];
    PhiFact=zeros(1,N_Var_Phi);
    for i=1:N_Var_Phi
        PhiFact(i)=360/VarRange(i);
    end
    % Parameters of von-Mises kernel
    I0=1/(2*pi*besseli(0,nu)); % The first kind modified Bessel function of order 0.

    for N_Var=1:N_Var_Phi
        PhiPeriodic=PhiFact(N_Var);
        Name_N=strcat('Cla_N_',num2str(VarRange(N_Var)));
        Name_M=strcat('Cla_M_',num2str(VarRange(N_Var)));
        Name_Dis=strcat('Dis_',num2str(VarRange(N_Var)));
        Cla_N=eval(Name_N);
        Cla_M=eval(Name_M);
        Dis=eval(Name_Dis);
        Dis_pi=Dis/180*pi;
        Softmax=zeros(Cla_N,Cla_M);
        for N_M=1:Cla_M
            Namenet=strcat(Material,'_net_',char(VarPhiList(N_Var)),'_',num2str(N_M));
            net=eval(Namenet);
            if strcmp(Parallel,'Yes') 
                y_pred=net(InputPara,'useParallel','yes');
            else 
                y_pred=net(InputPara);
            end 
            Softmax(:,N_M)=y_pred;
            clear net Namenet y_pred
        end
        Density=zeros(Cla_M,Cla_N);
        for Den_M=1:Cla_M
            for Den_N=1:Cla_N
                Den_Temp=0;
                for Dis_M=1:Cla_M
                    for Dis_N=1:Cla_N
                        Den_Temp=Den_Temp+Softmax(Dis_N,Dis_M)*von_Mises(I0,nu,Dis_pi(Den_M,Den_N)-Dis_pi(Dis_M,Dis_N));
                    end
                end
                Density(Den_M,Den_N)=Den_Temp;
            end
        end
        Den_Max=max(max(Density));
        [Posi_M,Posi_N]=find(Density==Den_Max);
        Phi_Temp=(Dis(Posi_M(1),Posi_N(1))-180)/PhiPeriodic;
        NamePhi=char(VarPhiList(N_Var));
        eval([NamePhi,'=Phi_Temp;']);
        clear NamePhi Phi_Temp
    end  

    Phi0_Phi=Phi1_Phi+Phi0_Phi1;
    if strcmp(Ref_Phi,'Phi')
        DeltPhi=Angle/180*pi;
        DeltPhi1=(Phi1_Phi+Angle)/180*pi;
        DeltPhi0=(Phi0_Phi+Angle)/180*pi;
    elseif strcmp(Ref_Phi,'Phi1')
        DeltPhi=(-Phi1_Phi+Angle)/180*pi;
        DeltPhi1=Angle/180*pi;
        DeltPhi0=(Phi0_Phi1+Angle)/180*pi;
    elseif strcmp(Ref_Phi,'Phi0')
        DeltPhi=(-Phi0_Phi+Angle)/180*pi;
        DeltPhi1=(-Phi0_Phi1+Angle)/180*pi;
        DeltPhi0=Angle/180*pi;
    end    

    %% Regression for modulus
    VarModulusList=["Gamma0","T","R","T0","T1","R0","R1"];
    N_Var_Modulus=length(VarModulusList);

    for N_Var=1:N_Var_Modulus
        Namenet=strcat(Material,'_net_',char(VarModulusList(N_Var)));
        net=eval(Namenet);
        if strcmp(Parallel,'Yes') 
            y_pred=net(InputPara,'useParallel','yes');
        else
            y_pred=net(InputPara);
        end 
        NameVar=char(VarModulusList(N_Var));
        eval([NameVar,'=y_pred;']);
        clear net Namenet y_pred NameVar
    end

    %% Define strain
    Num_epsilon=100;
    List_epsilon=zeros(Num_epsilon,1);
    List_epsilon(:,1)=linspace(0,epsilon_max,Num_epsilon)'; 
    if strcmp(Contribution,'Yes')
        NameFig=strcat('Gamma_',ModeDeform,'_on_',Material,'_[',num2str(n_surface(1),2),...
            ',',num2str(n_surface(2),2),',',num2str(n_surface(3),2),'] with reference_',...
            Ref_Phi,'=',num2str(Angle),'_degree and nu=',num2str(nu),'_Contri.png');
    else
        NameFig=strcat('Gamma_',ModeDeform,'_on_',Material,'_[',num2str(n_surface(1),2),...
            ',',num2str(n_surface(2),2),',',num2str(n_surface(3),2),'] with reference_',...
            Ref_Phi,'=',num2str(Angle),'_degree and nu=',num2str(nu),'.png');
    end
    Temp=zeros(Num_epsilon,4);
    Temp(:,1)=List_epsilon(:,1);

    %% Evaluation of Gamma with given strains
    if strcmp(ModeDeform,'Uniaxial_Compression')
        for N_epsilon=1:Num_epsilon
            t=-List_epsilon(N_epsilon,1)/2;
            r=List_epsilon(N_epsilon,1)/2;
            phi=pi/2;
            Temp(N_epsilon,2:4)=Gamma_Contribution(Gamma0,T,R,T0,R0,T1,R1,DeltPhi,DeltPhi0,DeltPhi1,t,r,phi);
        end
    elseif strcmp(ModeDeform,'Uniaxial_Tension')
        for N_epsilon=1:Num_epsilon
            t=List_epsilon(N_epsilon,1)/2;
            r=List_epsilon(N_epsilon,1)/2;
            phi=0;
            Temp(N_epsilon,2:4)=Gamma_Contribution(Gamma0,T,R,T0,R0,T1,R1,DeltPhi,DeltPhi0,DeltPhi1,t,r,phi);
        end
    elseif strcmp(ModeDeform,'Biaxial_Compression')
        for N_epsilon=1:Num_epsilon
            t=-List_epsilon(N_epsilon,1);
            r=0;
            phi=0;
            Temp(N_epsilon,2:4)=Gamma_Contribution(Gamma0,T,R,T0,R0,T1,R1,DeltPhi,DeltPhi0,DeltPhi1,t,r,phi);
        end
    elseif strcmp(ModeDeform,'Biaxial_Tension')
        for N_epsilon=1:Num_epsilon
            t=List_epsilon(N_epsilon,1);
            r=0;
            phi=0;
            Temp(N_epsilon,2:4)=Gamma_Contribution(Gamma0,T,R,T0,R0,T1,R1,DeltPhi,DeltPhi0,DeltPhi1,t,r,phi);
        end
    elseif strcmp(ModeDeform,'Positive_Shear')
        for N_epsilon=1:Num_epsilon
            t=0;
            r=List_epsilon(N_epsilon,1);
            phi=-pi/4;    
            Temp(N_epsilon,2:4)=Gamma_Contribution(Gamma0,T,R,T0,R0,T1,R1,DeltPhi,DeltPhi0,DeltPhi1,t,r,phi);
        end
    elseif strcmp(ModeDeform,'Negative_Shear')
        for N_epsilon=1:Num_epsilon
            t=0;
            r=List_epsilon(N_epsilon,1);
            phi=pi/4;    
            Temp(N_epsilon,2:4)=Gamma_Contribution(Gamma0,T,R,T0,R0,T1,R1,DeltPhi,DeltPhi0,DeltPhi1,t,r,phi);
        end
    end

    FigureOut(Temp,NameFig,epsilon_max,Contribution);
    Strain_Magnitude=Temp(:,1);
    Gamma_Total=Temp(:,2);
    Gamma_Stress=Temp(:,3);
    Gamma_Stiffness=Temp(:,4);
end

%% Function to calculate Gamma
function Result=Gamma_Contribution(Gamma0,T,R,T0,R0,T1,R1,DeltPhi,DeltPhi0,DeltPhi1,t,r,phi)
    Result=zeros(1,3);
    % Total excess energy
    Result(1,1)=Gamma0+2*T*t+2*R*r*cos(2*(DeltPhi-phi))+4*T1*t^2+...
        8*R1*cos(2*(DeltPhi1-phi))*t*r+2*(R0*cos(4*(DeltPhi0-phi))+T0)*r^2;
    % Contribution of residual stress
    Result(1,2)=Gamma0+2*T*t+2*R*r*cos(2*(DeltPhi-phi));
    % Contribution of excess stiffness
    Result(1,3)=Gamma0+4*T1*t^2+8*R1*cos(2*(DeltPhi1-phi))*t*r+...
        2*(R0*cos(4*(DeltPhi0-phi))+T0)*r^2;
    
    % Normalized the Gamma by Gamma0
    Result=Result/Gamma0;
end

%% Function to transfert the surface normal vector in the standard inverse pole figure
function n_surface=SIPF(n_surface,T_sym)
    n_surface=n_surface/norm(n_surface);
    SQRT2=sqrt(2)/2;
    SQRT3=sqrt(3)/3;
    inter=[1,SQRT2,SQRT3;0,SQRT2,SQRT3;0,0,SQRT3]\n_surface';
    symm=0;
    n_surf=n_surface;
    while((inter(1)<-10^-12 || inter(2)<-10^-12 || inter(3)<-10^-12) && symm<48)
        symm=symm+1;
        n_surf=n_surface*T_sym{symm};
        inter=[1,SQRT2,SQRT3;0,SQRT2,SQRT3;0,0,SQRT3]\n_surf';
    end
    n_surface=n_surf;
end

%% von-Mises kernel
function kappa=von_Mises(I0,nu,theta)
    kappa=I0*exp(nu*cos(theta));
end

%% Function for figure
function []=FigureOut(Temp,NameFig,epsilon_max,Contribution)
    Lim_Y(1)=min(min(Temp(:,2:4)));
    Lim_Y(2)=max(max(Temp(:,2:4)));
    InterVal=Lim_Y(2)-Lim_Y(1);
    figure(1);
    if strcmp(Contribution,'Yes')
        p1=plot(Temp(:,1),Temp(:,2),'-k',Temp(:,1),Temp(:,3),'--r',Temp(:,1),Temp(:,4),'-.b','LineWidth',3);
        legend(p1,{'Total \Gamma','\Gamma from \bf{\Sigma}_{\alpha\beta}^0','\Gamma from \bf{C}_{\alpha\beta\kappa\lambda}^S'},'FontSize',32)
        legend('boxoff');
    else
        p1=plot(Temp(:,1),Temp(:,2),'-k','LineWidth',3);
        legend(p1,{'Total \Gamma'},'FontSize',32);
        legend('boxoff');
    end
    hold on;
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 30 24])
    ax=gca;
    set(ax,'FontSize',32,'fontname','Times','linewidth',2,'TickDir','out','Position',[0.2,0.2,0.75,0.75]);
    set(ax,'box','off','color','none');
    bx = axes('Position',get(ax,'Position'),'box','on','FontSize',32,'linewidth',2,'XAxisLocation','top','xtick',[],'YAxisLocation','right','ytick',[]);
    set(bx,'box','off','color','none');
    axes(ax);
    axes(bx);
    linkaxes([ax bx],'xy');
    ax.XLim=[0,epsilon_max];
    ax.YLim=[Lim_Y(1)-InterVal*0.1,Lim_Y(2)+InterVal*0.1];
    xlabel(ax,char(949));        %title of axis x
    % Normalized the Gamma by Gamma0
    ylabel(ax,'\Gamma / \Gamma_0');        %title of axis y
    print(NameFig,'-dpng','-r300')
    close all;
end
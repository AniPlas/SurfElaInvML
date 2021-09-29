% Function to create the database of invariants from the
% semi-analytical calculation presented in Cartesian frame. 
function []=Create_Database(Cla_N_90,Cla_M_90,Cla_N_180,Cla_M_180)
    List_Material=["Ag","Al","Au","Cu","Ni","Pd","Pt"];
    N_Material=length(List_Material);
    VarList=["Gamma0","T","R","T0","T1","R0","R1","Phi1_Phi","Phi0_Phi1"];
    [~,N_Var]=size(VarList);
    
    Dis_UniN_90=360/Cla_N_90;
    Dis_UniM_90=Dis_UniN_90/Cla_M_90;
    Dis_90=zeros(Cla_M_90,Cla_N_90);
    for i=1:Cla_N_90
        for j=1:Cla_M_90
            Dis_90(j,i)=(j-1)*Dis_UniM_90+(i-1)*Dis_UniN_90;
        end
    end

    Dis_UniN_180=360/Cla_N_180;
    Dis_UniM_180=Dis_UniN_180/Cla_M_180;
    Dis_180=zeros(Cla_M_180,Cla_N_180);
    for i=1:Cla_N_180
        for j=1:Cla_M_180
            Dis_180(j,i)=(j-1)*Dis_UniM_180+(i-1)*Dis_UniN_180;
        end
    end

    save('Database_FCC_FreeSurface.mat','Dis_90','Dis_180');
    load('Data_SemiAnalytic.mat','T_sym');
    save('Database_FCC_FreeSurface.mat','T_sym','-append');
    for N_Mat=1:N_Material      
        Mat=char(List_Material(N_Mat));
        Name_Gamma0=strcat(Mat,'_Gamma0');
        Name_Gamma1=strcat(Mat,'_Gamma1');
        Name_Gamma2=strcat(Mat,'_Gamma2');
        Name_Vec_z_cry=strcat(Mat,'_Vec_z_cry');
        
        load('Data_SemiAnalytic.mat',Name_Gamma0,Name_Gamma1,Name_Gamma2,Name_Vec_z_cry);

        Gamma0=eval(Name_Gamma0);
        Gamma1=eval(Name_Gamma1);
        Gamma2=eval(Name_Gamma2);
        Vec_z_cry=eval(Name_Vec_z_cry);

        [N_Symm,~]=size(Vec_z_cry);
        ML_Results=zeros(N_Symm,2+N_Var);
        
        % Gamma0 surface energy

        % Gamma1 surface stress 
        T=zeros(N_Symm,1);
        R=zeros(N_Symm,1);
        Phi=zeros(N_Symm,1);
        for i=1:N_Symm
            T(i)=(Gamma1{i}(1,1)+Gamma1{i}(2,2))/2;
            R(i)=sqrt(((Gamma1{i}(1,1)-Gamma1{i}(2,2))/2)^2+(Gamma1{i}(1,2))^2);
            sin_2Phi=Gamma1{i}(1,2);
            cos_2Phi=(Gamma1{i}(1,1)-Gamma1{i}(2,2))/2;
            Phi(i)=atan2(sin_2Phi,cos_2Phi)/2/pi*180;
            if Phi(i)<0
                Phi(i)=Phi(i)+360;
            end
        end

        % Gamma2 surface stiffness
        T0=zeros(N_Symm,1);
        T1=zeros(N_Symm,1);
        R0=zeros(N_Symm,1);
        R1=zeros(N_Symm,1);
        Phi0=zeros(N_Symm,1);
        Phi1=zeros(N_Symm,1);
        for i=1:N_Symm
            T0(i)=(Gamma2{i}(1,1,1,1)-2*Gamma2{i}(1,1,2,2)+4*Gamma2{i}(1,2,1,2)+Gamma2{i}(2,2,2,2))/8;
            T1(i)=(Gamma2{i}(1,1,1,1)+2*Gamma2{i}(1,1,2,2)+Gamma2{i}(2,2,2,2))/8;
            R0(i)=sqrt(((Gamma2{i}(1,1,1,2)-Gamma2{i}(2,2,1,2))/2)^2+((Gamma2{i}(1,1,1,1)-2*Gamma2{i}(1,1,2,2)-4*Gamma2{i}(1,2,1,2)+Gamma2{i}(2,2,2,2))/8)^2);
            R1(i)=sqrt(((Gamma2{i}(1,1,1,2)+Gamma2{i}(2,2,1,2))/4)^2+((Gamma2{i}(1,1,1,1)-Gamma2{i}(2,2,2,2))/8)^2);
            sin_4Phi0=(Gamma2{i}(1,1,1,2)-Gamma2{i}(2,2,1,2))/2;
            cos_4Phi0=(Gamma2{i}(1,1,1,1)-2*Gamma2{i}(1,1,2,2)-4*Gamma2{i}(1,2,1,2)+Gamma2{i}(2,2,2,2))/8;
            Phi0(i)=atan2(sin_4Phi0,cos_4Phi0)/4/pi*180;
            if Phi0(i)<0
                Phi0(i)=Phi0(i)+360;
            end
            sin_2Phi1=(Gamma2{i}(1,1,1,2)+Gamma2{i}(2,2,1,2))/4;
            cos_2Phi1=(Gamma2{i}(1,1,1,1)-Gamma2{i}(2,2,2,2))/8;
            Phi1(i)=atan2(sin_2Phi1,cos_2Phi1)/2/pi*180;
            if Phi1(i)<0
                Phi1(i)=Phi1(i)+360;
            end
        end

        Phi1_Phi=zeros(N_Symm,1);
        Phi0_Phi1=zeros(N_Symm,1);
        for i=1:N_Symm
            Phi1_Phi(i)=PeriodicAngle180(Phi1(i),Phi(i));
            Phi0_Phi1(i)=PeriodicAngle90(Phi0(i),Phi1(i));
        end

        ML_Results(:,1)=acos(Vec_z_cry(:,3))/pi*180;
        ML_Results(:,2)=atan(Vec_z_cry(:,2)./Vec_z_cry(:,1))/pi*180;

        for Var=1:N_Var
            ML_Results(:,2+Var)=eval(VarList(Var));
        end
        
%         % Make random orders
% %         Order=randperm(N_Symm);
%         load('Data_SemiAnalytic.mat','Order');
%         ML_Results_Random=zeros(N_Symm,2+N_Var);
%         for Ran_Order=1:N_Symm
%             ML_Results_Random(Ran_Order,:)=ML_Results(Order(Ran_Order),:);
%         end
        ML_Results_Random=ML_Results;

        % Create target matrix
        ML_target_Phi1_Phi=zeros(N_Symm,Cla_N_180,Cla_M_180); 
        ML_target_Phi0_Phi1=zeros(N_Symm,Cla_N_90,Cla_M_90);
        for i=1:N_Symm
            for j=1:Cla_M_90
                ML_target_Phi0_Phi1(i,:,j)=TargetMap(ML_Results_Random(i,11)*4+180,Dis_90(j,:),Cla_N_90);
            end
            for j=1:Cla_M_180
                ML_target_Phi1_Phi(i,:,j)=TargetMap(ML_Results_Random(i,10)*2+180,Dis_180(j,:),Cla_N_180);
            end
        end
        
        % Output text
        InputFac=ML_Results_Random(:,1:2)';
        Name_InputFac=strcat(Mat,'_InputFac');
        eval([Name_InputFac,'=InputFac;']);
        save('Database_FCC_FreeSurface.mat',Name_InputFac,'-append');

        for j=1:Cla_M_90
            Dataname=strcat(Mat,'_Phi0_Phi1_',num2str(j)); 
            Temp=ML_target_Phi0_Phi1(:,1:Cla_N_90,j)';
            eval([Dataname,'=Temp;']);
            save('Database_FCC_FreeSurface.mat',Dataname,'-append');
        end

        for j=1:Cla_M_180
            Dataname=strcat(Mat,'_Phi1_Phi_',num2str(j)); 
            Temp=ML_target_Phi1_Phi(:,1:Cla_N_180,j)';
            eval([Dataname,'=Temp;']);
            save('Database_FCC_FreeSurface.mat',Dataname,'-append');
        end

        for Var=1:N_Var
            Temp=ML_Results_Random(:,2+Var)';
            NameTarget=strcat(Mat,'_Target_',char(VarList(Var))); 
            eval([NameTarget,'=Temp;'])
            save('Database_FCC_FreeSurface.mat',NameTarget,'-append');
        end
        
        clear DataName Gamma0 Gamma1 Gamma2 Poten Vec_z_cry ...
            Name_Gamma0 Name_Gamma1 Name_Gamma2 Name_Vec_z_cry Name_Poten
    end
end

%%
function Angle=PeriodicAngle180(Phi_tar,Phi_ref)
    inter=abs(Phi_tar-Phi_ref);
    if inter>180
        inter=360-inter;
    end
    inter3=cross([cos(Phi_tar/180*pi),sin(Phi_tar/180*pi),0],[cos(Phi_ref/180*pi),sin(Phi_ref/180*pi),0]);
    if inter>90
        Angle=sign(inter3(3))*(180-inter);
    else
        Angle=-sign(inter3(3))*inter;
    end
end

%%
function Angle=PeriodicAngle90(Phi_tar,Phi_ref)
    inter=abs(Phi_tar-Phi_ref);
    if inter>180
        inter=360-inter;
    end
    if inter>90
        inter=180-inter;
    end
    if inter>45
        Phi_tar=Phi_tar+90;
        if Phi_tar>=360
            Phi_tar=Phi_tar-360;
        end
    end
    inter=abs(Phi_tar-Phi_ref);
    if inter>180
        inter=360-inter;
    end
    inter3=cross([cos(Phi_tar/180*pi),sin(Phi_tar/180*pi),0],[cos(Phi_ref/180*pi),sin(Phi_ref/180*pi),0]);
    if inter>90
        Angle=sign(inter3(3))*(180-inter);
    else
        Angle=-sign(inter3(3))*inter;
    end
end

%%
function ML_target=TargetMap(ML_Results,Dis,Cla_N)
    ML_target=zeros(1,Cla_N);
    [~,Posi]=min(abs(Dis-ML_Results));
    ML_target(Posi)=1;
end
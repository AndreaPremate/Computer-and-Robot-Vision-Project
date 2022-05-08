function PF
    

    % Close all windows and reset workspace
    clear all;
    close all;
    %% inizializzazione
    % Robot & Environment configuration
    filename = 'simulated_data4t.txt';        
    baseline = 0.5;                   % wheels base / interaxis
    focal = 600;                     % focal length of the camera (in pixels)
    camera_height = 10;               % how far is the camera w.r.t. the robot ground plane
    pTpDistance = 0.6;                  % distance between two points on the robot
    
    % Read & parse data from the input file
    DATA=read_input_file(filename);    
    odometry  = [DATA(:,3),DATA(:,4)];    
    camera_readings1 = [DATA(:,11),DATA(:,12),DATA(:,13),DATA(:,14)];
    camera_readings2 = [DATA(:,15),DATA(:,16),DATA(:,17),DATA(:,18)];
    camera_readings3 = [DATA(:,19),DATA(:,20),DATA(:,21),DATA(:,22)];
    camera_readings4 = [DATA(:,23),DATA(:,24),DATA(:,25),DATA(:,26)];
    
    steps = size(odometry,1);
    
    % world dimensions (meter)
    lenght = 20;
    width  = 20;
    
    % number of particles
    number_of_particles = 300;
    % con solamente 300 particle e mondo 20x20 si ottengono entrambi i percorsi nella
    % maggior parte dei casi grazie all'approccio della temperatura (con gaussian_max).
    % usando 500 particelle si ottengono entrambi i percorsi praticamente
    % nella totalità dei casi (con gaussian_max)
    % 
    % con gaussian mixture nella parte di sovrapposizione di letture molto spesso il
    % cluster non sovrapposto viene azzerato

    % initialize the filter with RANDOM samples
    particle_set_a = [lenght * rand(1, number_of_particles) - lenght/2; ...
                      width * rand(1, number_of_particles) - width/2; ...
                      2 * pi * rand(1, number_of_particles)];
    particle_set_b   = zeros(3, number_of_particles);
    particle_weights = repmat(1/number_of_particles, number_of_particles)';
    
    show_particles(particle_set_a);
    temperature = 16;

    for i=1:steps-1
        
        %% prediction
        for particle=1:number_of_particles
            particle_set_b(:, particle)=execute_prediction(particle_set_a(:,particle),odometry(i, :), baseline, temperature);            
        end
        
        %% plot
        figure(2)
        % plot dei percorsi
        clf; title('CAMERA READINGS'); set(gca,'YDir','reverse'); hold on        
        plot(camera_readings1(:,1),camera_readings1(:,2),'b*');
        plot(camera_readings2(:,1),camera_readings2(:,2),'g*');
        plot(camera_readings3(:,1),camera_readings3(:,2),'y*');
        plot(camera_readings4(:,1),camera_readings4(:,2),'m*');
        % plot delle prediction delle particles
        plot((focal/camera_height)*particle_set_b(1, :), (focal/camera_height) * -particle_set_b(2, :), 'c*');
        % plot delle letture da camera al tempo i
        plot(camera_readings1(i,1),camera_readings1(i,2),'s','MarkerSize',10,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6]);
        plot(camera_readings2(i,1),camera_readings2(i,2),'s','MarkerSize',10,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6]);
        plot(camera_readings3(i,1),camera_readings3(i,2),'s','MarkerSize',10,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6]);
        plot(camera_readings4(i,1),camera_readings4(i,2),'s','MarkerSize',10,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6]);
        
        % se si conosce il numero di oggetti che sono ripresi dalla camera
        % (nel nostro caso 2 considerando la lettura sfalsata) si può
        % calcolare il centroide di ciascun cluster usando il k-means per
        % ottenere i due stati più probabili.
        [clust_index, centroids] = kmeans(particle_set_b(1:2,:)', 2);
        centroids = centroids';
        plot((focal/camera_height)*centroids(1, :), (focal/camera_height) * -centroids(2, :), 'ko','MarkerFaceColor',[0 0 0]);
        
        % la legenda rimpicciolisce e deforma il plot,
        % decommentare solo se ne si ha bisogno:
%         legend({'cam1','cam2', 'cam3', 'cam4', 'Particles', ...
%                 'cam1234_t','', '', '', 'centroids'},'Location', 'northeastoutside','Orientation','vertical','NumColumns',1);

        hold off

        %% weight
        for particle=1:number_of_particles
            particle_weights(particle) = weight_particle(particle_set_b(:,particle),camera_readings1(i, :), ...
            camera_readings2(i, :),camera_readings3(i, :),camera_readings4(i, :),focal,camera_height, pTpDistance, 'gaussian_max');
        end
        
        %% resapling
        particle_set_a = resample(particle_set_b, particle_weights,'systematic');
        
        show_particles(particle_set_a);
        particle_weights=zeros(number_of_particles);

        if temperature > 1
            temperature = temperature - 1;
        end
%         pause(0.01);
    end
end


function w = weight_particle(particle_state,camera_readings1,camera_readings2,...
                                            camera_readings3,camera_readings4, focal, camera_height,pTpDistance, method)

    x = particle_state(1);
    y = particle_state(2);
    theta = particle_state(3);
    u1 = (focal/camera_height) * x;
    v1 = (focal/camera_height) * (-y);
    u2 = (focal/camera_height) * (cos(theta)*pTpDistance + x);
    v2 = (focal/camera_height) * (sin(theta)*pTpDistance - y);
   
    % ottengo così la particella in coordinate camera e posso confrontarla
    % con le letture della camera; per dare un peso alla particella è
    % possibile scegliere tra due opzioni: gaussian_mixture e gaussian_max
    particle_cam_view = [u1;v1;u2;v2];
    
    mu_P1 = [];
    mu_P2 = [];
    sigma_P1 = [900 900]; % diagonale cov
    sigma_P2 = [900 900]; % diagonale cov
    tot_read = 0;
    for camera_obs=[camera_readings1; camera_readings2; camera_readings3; camera_readings4]'
        if ~isnan(camera_obs)
            % 50 cm (coord mondo) in coord camera equivale a 30 pixel -> 30^2 (varianza) = 900
            % per dare un peso alla particle mi baso su entrambi punti riconoscibili del robot,
            % valutandoli equamente importanti (vedi riga 255 o 265)
            mu_P1 = [mu_P1 ; camera_obs(1) camera_obs(2)];
            mu_P2 = [mu_P2 ; camera_obs(3) camera_obs(4)];
            tot_read = tot_read + 1;
        end
    end
    if strcmp(method, 'gaussian_mixture')
        % creo le misture di gaussiane per i punti P1 e P2
        gmP1 = gmdistribution(mu_P1,sigma_P1);  %Covariance matrices are diagonal and the same across components.
        gmP2 = gmdistribution(mu_P2,sigma_P2);  %Covariance matrices are diagonal and the same across components.

        % peso allo stesso modo P1 e P2
        w = pdf(gmP1, [particle_cam_view(1) particle_cam_view(2)]) + pdf(gmP2, [particle_cam_view(3) particle_cam_view(4)]);
    end

    if strcmp(method, 'gaussian_max')
        max = -inf;
        ii = 1;
        while ii <= tot_read
            % considero una lettura di camera alla volta
            gmP1 = gmdistribution([mu_P1(ii,1) mu_P1(ii,2)], sigma_P1); % gaussian mixture con 1 sola gaussiana alla volta (per comodità e mantenere simmetria nel codice)
            gmP2 = gmdistribution([mu_P2(ii,1) mu_P2(ii,2)], sigma_P2); % gaussian mixture con 1 sola gaussiana alla volta
            tmp = pdf(gmP1, [particle_cam_view(1) particle_cam_view(2)]) + pdf(gmP2, [particle_cam_view(3) particle_cam_view(4)]);
            % trovo il massimo considerando la somma delle probabilità di
            % P1 e P2 per ogni lettura
            if tmp > max
                max = tmp;
            end
            ii = ii +1;
        end
        w = max;
    end

end


function prediction=execute_prediction(particle_state,odometry,baseline, temperature)
    % Mi baso sull'equazionde di stato per calcolare la prediction di
    % ciascuna particle. 
    % Allo stato viene sommato rumore gaussiano di intensità
    % decrescente col procedere del tempo (la temperatura iniziale è
    % impostata a 16 e scende di 1 ad ogni istante di tempo, fino ad 
    % arrivare ai valori fissi [randn*0.1; randn*0.1; randn*0.1] ).
    x = particle_state(1);
    y = particle_state(2);
    theta = particle_state(3);
    sdx = odometry(1);
    ssx = odometry(2);
    if ssx == sdx
        new_x = x + ssx*cos(theta);
        new_y = y - ssx*sin(theta);
        new_theta = theta;
    else 
        m = baseline * sdx / (ssx - sdx);
        alpha = (ssx - sdx) / baseline;
        new_x = x + cos(theta)*sin(alpha)*(m + baseline/2) + ...
            sin(theta)*(cos(alpha)*(m + baseline/2) - m - baseline/2);
        new_y = y - sin(theta)*sin(alpha)*(m + baseline/2) + ...
        cos(theta)*(cos(alpha)*(m + baseline/2) - m - baseline/2);
        new_theta = theta + alpha;                             
    end
    prediction = [new_x; new_y; new_theta] + [randn*0.1; randn*0.1; randn*0.1]* temperature;
end
    

function show_particles(particle_set)
% Do not edit this function 
    figure(1);      
    clf(1);
    title('PARTICLES');
    hold on;        
        plot(particle_set([1],:),particle_set([2],:),'k*')
        arrow=0.5;
        %
        % line([particle_set(1,:); particle_set(1,:)+arrow*(cos(particle_set(3,:).*-1))], [particle_set(2,:); particle_set(2,:)+arrow*(-sin(particle_set(3,:).*-1))],'Color','b');
        %
        % l'angolo per come l'ho definito io nel sistema mondo è l'opposto
        % rispetto alla tipica convenzione senso positivo -> antiorario
        % quindi per visualizzarlo bisogna cambiare di segno la parte del
        % seno. Infatti cos(-a) = cos(a) quindi rimane invariato, mentre
        % sin(-a) = -sin(a).
        % versione adattata al mio angolo:
        line([particle_set(1,:); particle_set(1,:)+arrow*(cos(particle_set(3,:).*-1))], [particle_set(2,:); particle_set(2,:)+arrow*(-sin(particle_set(3,:)))],'Color','b');  
    hold off;
    
end

function simulated_data = read_input_file(filename)
% Do not edit this function 
    
    simulated_data = dlmread(filename,';');
    
end


function particle_set = resample(particle_set,weights,method)
    % ottengo una distribuzione di probabilità
    weights = weights / sum(weights);
    
    w_cumsum = cumsum(weights);
    N = length(weights);
    % sampling
    if strcmp(method, 'stratified')
        u =([0:N-1]+ rand(1,N))/N;
    else 
        if strcmp(method, 'systematic')
            u =([0:N-1]+ rand(1))/N;
        end
    end
    
    ind = zeros(1,N);
    j = 1;
    for i = 1:N
        while (w_cumsum(j)<u(i))
            j = j+1;
        end
        ind(i) = j ;
    end
    particle_set = particle_set(:,ind);
end
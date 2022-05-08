function EKF

    %% Close all windows and reset workspace
    clear all;
    close all;

    %% Robot & Environment configuration
    filename = 'simulated_data.txt';    % better not to edit the filename for the simulated data
    baseline = 0.5;                     % distance between the contact points of the robot wheels [m]
    focal = 600;                       % focal length of the camera [pixel]
    camera_height = 10;                 % how far is the camera w.r.t. the world plane [m]
    pTpDistance = 0.6;                    % distance between the two observed points on the robot [m]

    %% Read & parse data from the input file
    DATA=read_input_file(filename);    
    odometry  = [DATA(:,3),DATA(:,4)];    
    camera_readings = [DATA(:,11),DATA(:,12),DATA(:,13),DATA(:,14)];

    steps = size(odometry,1); 
    kalmanstate = zeros(steps,3);       % memory for the 3 state variables, for each timestamp
    kalmanstatecov = zeros(3,3,steps);  % memory for the covariance of the 3 state variables, for each timestamp
    prediction = zeros(steps,3);        % memory for the 3 predicted state variables, for each timestamp
    predictedcov = zeros(3,3,steps);    % memory for the covariance of the 3 predicted state variables, for each timestamp

    %% Initializations 

    x_start_P1 = camera_readings(1,1)*camera_height/focal;
    y_start_P1 = -camera_readings(1,2)*camera_height/focal;
    x_start_P2 = camera_readings(1,3)*camera_height/focal;
    y_start_P2 = -camera_readings(1,4)*camera_height/focal;
    a = [x_start_P2 - x_start_P1; y_start_P2 - y_start_P1];
    b = [1 ; 0];
    theta_start = atan2(det([a, b]), dot(a,b));     % angolo iniziale (il det è proporzionale al seno e il dot prod al coseno)

    kalmanstate(1,:) = [x_start_P1, y_start_P1, theta_start];       % initial state estimate
    % Basandosi sulla prima osservazione proveniente dalla camera imposto
    % lo stato iniziale.

    kalmanstatecov(:,:,1) = [0.01 0 0; 0 0.01 0; 0 0 0.00761544];      % covariance of the initial state estimate, a (3,3) matrix; 

    %% configuration of uncertainties
    R(:,:) = 0.001*eye(3);     % covariance of the noise acting on the state, a (3,3) matrix;

    Q(:,:) = 36*eye(4);     % covariance of the noise acting on the measure, a (4,4) matrix; 


    % Create the symbolic matrices representing the G and H matrices.
    [H,symb_expected_camera_readings] = calculateSymbolicH_AND_pred_observation;
    
    %% batch-simulation of the evolution of the system
    for i=1:steps-1
        if (odometry(i+1,1) ~= odometry(i+1,1))
            [G,symb_predicted_state] = calculateSymbolicGRot_AND_pred_state;
        else
            [G,symb_predicted_state] = calculateSymbolicGRet_AND_pred_state;
        end

        % Print the current status of the system in the command window
        fprintf('Filtering step %d/%d; Current uncertainty (x,y,ang): %0.6f %0.6f %0.6f\n',i,steps, kalmanstatecov(1,1,i), kalmanstatecov(2,2,i),kalmanstatecov(3,3,i)); 
        [kalmanstate(i+1,:), kalmanstatecov(:,:,i+1), prediction(i+1,:), predictedcov(:,:,i+1)] = execute_kalman_step(kalmanstate(i,:), ...
            kalmanstatecov(:, :, i), odometry(i+1,:), camera_readings(i+1,:), ...
            baseline, R,Q, focal, camera_height, pTpDistance, G, H, symb_predicted_state, symb_expected_camera_readings);
    end


    %% Plot the results
    figure(3);
    hold on;
        plot(kalmanstate(:,1), kalmanstate(:,2),'g+');      
        plot(prediction(:,1), prediction(:,2),'c.');

        % plot where the robot points would be on the ground plane for a given measurement (if used as a measurement, this expression would be an inverse sensor model)
        plot(camera_readings(:,1)*camera_height/focal, -camera_readings(:,2)*camera_height/focal,'m*');

        % plot of state uncertainty
        for i=1:1:steps
            plot_gaussian_ellipsoid(kalmanstate(i,1:2),kalmanstatecov(1:2,1:2,i),3.0); 
            % l'ellissi rappresenta 3 std quindi ci si aspetta che la quasi
            % totalità delle osservazioni della camera rientri
            % nell'incertezza definita da tali ellissi.
        end

        axis equal
    hold off; 

    figure(4);
    hold on;
        %plot(kalmanstate(:,1), kalmanstate(:,2),'g+');      
        line(kalmanstate(:,1), kalmanstate(:,2),'Color','b')

        % plot where the robot points would be on the ground plane for a given measurement (if used as a measurement, this expression would be an inverse sensor model)
        %plot(camera_readings(:,1)*camera_height/focal, -camera_readings(:,2)*camera_height/focal,'m*');
        line(camera_readings(:,1)*camera_height/focal, -camera_readings(:,2)*camera_height/focal,'Color','r');
        % plot of state uncertainty
       

        axis equal
    hold off; 

end



function simulated_data = read_input_file(filename) % Do not edit this function
simulated_data = dlmread(filename,';'); 
end



function [filtered_state, filtered_sigma, predicted_state, predicted_sigma]= execute_kalman_step...
    (current_state,current_state_sigma,odometry,camera_readings,baseline,R,Q,focal,camera_height,pTpDistance,G,H,symb_predicted_state,symb_expected_camera_readings)

    % memory for the expected camera readings
    expected_camera_readings = zeros(1,4);

    syms symb_x symb_y symb_theta symb_l symb_sdx symb_ssx        
    G = eval(subs(G,[symb_x,symb_y,symb_theta,symb_l,symb_sdx,symb_ssx],[current_state(1), current_state(2), ...
        current_state(3), baseline, odometry(1),odometry(2)]));

    syms symb_focal symb_pTpDistance symb_camera_height symb_x symb_y symb_theta
    H = eval(subs(H,[symb_focal,symb_pTpDistance,symb_camera_height,symb_x,symb_y,symb_theta],[focal, pTpDistance, camera_height, ...
        current_state(1), current_state(2),current_state(3)]));

    % Generate the expected new pose, based on the previous pose and the controls, i.e., the dead reckoning data

    predicted_state(1,:) = eval(subs(symb_predicted_state,[symb_ssx, symb_sdx, symb_l, symb_x, symb_y, symb_theta],[odometry(1),odometry(2), ...
                                baseline, current_state(1), current_state(2),current_state(3)]));
    predicted_sigma(:,:,1) = G * current_state_sigma * G' + R;

    % then integrate the expected new pose with the measurements
    K = predicted_sigma(:,:,1) * H' / (H * predicted_sigma(:,:,1) * H' + Q);

    expected_camera_readings = (eval(subs(symb_expected_camera_readings,[symb_focal symb_x symb_y symb_theta symb_pTpDistance symb_camera_height],[focal, ...
        current_state(1), current_state(2), current_state(3), pTpDistance, camera_height])))';
    filtered_state = predicted_state(1,:)' + K * (camera_readings - expected_camera_readings)';
    filtered_sigma = (eye(3) - K * H) * predicted_sigma(:,:,1);
end




function [H, observation]=calculateSymbolicH_AND_pred_observation
    % *** H MATRIX STEP, CALCULATION OF DERIVATIVES aka JACOBIANS ***
    syms symb_focal symb_x symb_y symb_theta symb_pTpDistance symb_camera_height
    u1 = (symb_focal/symb_camera_height) * symb_x;
    v1 = (symb_focal/symb_camera_height) * (-symb_y);
    u2 = (symb_focal/symb_camera_height) * (cos(symb_theta)*symb_pTpDistance + symb_x);
    v2 = (symb_focal/symb_camera_height) * (sin(symb_theta)*symb_pTpDistance - symb_y);
    
    observation = [u1;v1;u2;v2];
    H = jacobian(observation,[symb_x,symb_y,symb_theta]);
    
end

function [G, state]=calculateSymbolicGRot_AND_pred_state    % moto rotatorio
    % *** G MATRIX STEP, CALCULATION OF DERIVATIVES aka JACOBIANS ***
    syms symb_ssx symb_sdx symb_l symb_x symb_y symb_theta
    m = symb_l * symb_sdx / (symb_ssx - symb_sdx);
    alpha = (symb_ssx - symb_sdx) / symb_l;

    new_x = symb_x + cos(symb_theta)*sin(alpha)*(m + symb_l/2) + ...
        sin(symb_theta)*(cos(alpha)*(m + symb_l/2) - m - symb_l/2);
    
    new_y = symb_y - sin(symb_theta)*sin(alpha)*(m + symb_l/2) + ...
        cos(symb_theta)*(cos(alpha)*(m + symb_l/2) - m - symb_l/2);
    
    new_theta = symb_theta + alpha;
    state = [new_x;new_y;new_theta] ;
    G= jacobian(state,[symb_x,symb_y,symb_theta]);
end

function [G, state]=calculateSymbolicGRet_AND_pred_state    % moto rettilineo
    % *** G MATRIX STEP, CALCULATION OF DERIVATIVES aka JACOBIANS ***
    syms symb_ssx symb_l symb_x symb_y symb_theta

    new_x = symb_x + cos(symb_theta)*symb_ssx;
    new_y = symb_y - sin(symb_theta)*symb_ssx;
    new_theta = symb_theta;

    state = [new_x;new_y;new_theta] ;
    G= jacobian(state,[symb_x,symb_y,symb_theta]);
end
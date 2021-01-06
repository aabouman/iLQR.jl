module iLQR

using RigidBodyDynamics, RigidBodySim
using ForwardDiff
using StaticArrays
using GeometryTypes: Point

include("cost_functions.jl")
include("forward_pass.jl")

function backward_pass(nframes, nx)
    #nx is the number of states
    #nu is the numer of control inputs

    delta_x = x_des - x_nom;
    delta_u = u_des - u_nom;

    #initialize s quantities that we wish to calculate
    S_n_mat_arr = zeros((nx, nx, nframes));
    s_n_vec_arr = zeros(nx,nframes);
    S_n_mat_arr[:,:,nframes] = Q_N_mat; #TODO: define Q_N_mat and q_N_vec
    s_n_vec_arr[:,nframes] = q_N_vec;

    #Loop backwards through time to compute all s terms
    for i = (nframes-1):-1:1

        #obtain A & B matrices for this timestep
        A = A_cell_array{i+1};
        B = B_cell_array{i+1};


        S_np1_mat = S_n_mat_arr{i+1};
        minus_S_mat_dot = Q - S_np1_mat*B*inv(R)*B'*S_np1_mat + S_np1_mat*A + A'*S_np1_mat;
        S_n_mat = S_np1_mat + (minus_S_mat_dot)*dt;

        s_np1_vec = s_n_vec_arr{i+1};
        minus_s_vec_dot = -2*Q*delta_x(:,i+1) + (A'-S_np1_mat*B*inv(R)*B')*s_np1_vec + 2*S_np1_mat*B*delta_u(i+1);
        s_n_vec = s_np1_vec + (minus_s_vec_dot)*dt;



        S_n_mat_arr[:,:,i] = S_n_mat;
        s_n_vec_arr[:,i] = s_n_vec;



    end

end

function iLQR_main(args)
    body
end


end

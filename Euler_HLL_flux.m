function [Fn] = Euler_HLL_flux(U_l, U_r, p_l, p_r, normal, S_l, S_r, dim, use_HLLC)
% [Fn] = Euler_HLL_flux(U_l, U_r, p_l, p_r, normal, S_l, S_r, dim)
%   compute the Harten-Lax-van Leer (HLL) numerical flux on normal direction using conservative variables and pressure
%   (support any problem dimension)
%   (support any equation of state (EOS))
% 
% input:
%   U_l, U_r:   conservative variables [rho, momentum, total energy], numel == dim+2
%   p_l, p_r:   pressure, scalar
%   normal:     unit normal vector, numel == dim
%   S_l, S_r:   estimated left and right signal velocities, scalar
%   dim:        problem dimension, integer scalar
%   use_HLLC:   whether to use the HLLC flux, bool scalar
%   
% output:
%   Fn:         numerical flux on the normal direction, size = [dim+2, 1]
% 
% recommendation:
%   use the Einfeldt's esimate for S_l and S_r for better stability
% 
% Refs: 
%   [1] On the Choice of Wavespeeds for the HLLC Riemann Solver
%   [2] Entropy stable high order discontinuous Galerkin methods with suitable quarature rules for hyperbolic conservation laws
%   [3] On an inconsistency of the arithmetic-average signal speed estimate for HLL-type Riemann solvers
%   [4] Implementation of WENO schemes in compressible multicomponent flow problems
%   [5] On Godunov-type methods near low densities
%   [6] Simplified Second-Order Godunov-Type Methods
%   [7] A shock-stable modification of the HLLC Riemann solver with reduced numerical dissipation
% 
% author: 
%   Yue Wu (Email: yue_wu3@brown.edu, URL: https://yuewu2002.github.io/)

% pre-process the inputs
normal = reshape(normal(1:dim),[dim,1]);

rho_l = U_l(1); % density
m_l = reshape(U_l(2:dim+1), [dim,1]); % momentum
v_l = m_l / rho_l; % velocity
E_l = U_l(dim+2); % total energy per unit volume
vn_l = dot(v_l(:), normal(:)); % normal speed

rho_r = U_r(1); % density
m_r = reshape(U_r(2:dim+1), [dim,1]); % momentum
v_r = m_r / rho_r; % velocity
E_r = U_r(dim+2); % total energy per unit volume
vn_r = dot(v_r(:), normal(:)); % normal speed

% compute the HLL-famliy numerical flux
Fn = nan(dim+2,1); % declaration
if S_l >= 0.0
    % supersonic flow toward right
    % use all information from left

    % the exact flux
    Fn(1) = vn_l * rho_l;
    Fn(2:dim+1) = vn_l * m_l(:) + p_l * normal(:);
    Fn(dim+2) = vn_l * (E_l + p_l);

    return;
elseif S_r <= 0.0
    % supersonic flow toward left
    % use all information from right

    % the exact flux
    Fn(1) = vn_r * rho_r;
    Fn(2:dim+1) = vn_r * m_r(:) + p_r * normal(:);
    Fn(dim+2) = vn_r * (E_r + p_r);

    return;
else
    % subsonic
    % use information from both left and right
    % S_l < 0.0 < S_r

    rvn_l = vn_l - S_l; % velocity into the Riemann fan at the 1-wave
    rvn_r = vn_r - S_r; % velocity from the Riemann fan at the 3-wave

    if use_HLLC
        % compute the HLLC flux

        flow_l = rho_l * rvn_l; % mass flow into the Riemann fan at the 1-wave
        flow_r = rho_r * rvn_r; % mass flow from the Riemann fan at the 3-wave

        mnflux_l = vn_l * flow_l + p_l; % momentum flux into the Riemann fan at the 1-wave
        mnflux_r = vn_r * flow_r + p_r; % momentum flux from the Riemann fan at the 3-wave

        % estimated contact wave speed (two-shock assumption)
        % also the normal velocity between two shocks
        S_m = (mnflux_l - mnflux_r) / (flow_l - flow_r);

        if S_m > 0.0
            % to the left of the contact wave
            % using jump conditions of the 1-wave
            % Fn = F_l + S_l*(U_cl^* - U_cl)
            
            % auxiliary variables
            ts_l = ((vn_l - S_m) / (S_l - S_m)) * S_l;
            tsflow_l = ts_l * flow_l;
            vv_l = vn_l - ts_l;

            % the HLLC flux
            Fn(1) = vv_l * rho_l;
            Fn(2:dim+1) = vv_l * m_l(:) + (p_l + tsflow_l) * normal(:);
            Fn(dim+2) = vv_l * (E_l + p_l) + tsflow_l * S_m;

            return;
        elseif S_m < 0.0
            % to the right of the contact wave
            % using jump conditions of the 3-wave
            % Fn = F_r + S_r*(U_cr^* - U_cr)

            % auxiliary variables
            ts_r = ((vn_r - S_m) / (S_r - S_m)) * S_r;
            tsflow_r = ts_r * flow_r;
            vv_r = vn_r - ts_r;

            % the HLLC flux
            Fn(1) = vv_r * rho_r;
            Fn(2:dim+1) = vv_r * m_r(:) + (p_r + tsflow_r) * normal(:);
            Fn(dim+2) = vv_r * (E_r + p_r) + tsflow_r * S_m;

            return;
        else % S_m == 0.0
            % exactly at the contact wave
            % using symmetry
            
            % the HLLC flux
            Fn(1) = 0.0;
            Fn(2:dim+1) = (0.5 * (mnflux_l + mnflux_r)) * normal(:);
            Fn(dim+2) = 0.0;

            return;
        end

    else % use_HLLC == false
        % compute the HLL flux

        width = S_r - S_l; % width of the Riemann fan
        t_l = S_r / width; % weight of the left part (in [0,1])
        t_r = 1.0 - t_l; % weight of the right part (in [0,1])

        % auxiliary variables
        trvn_l = t_l * rvn_l;
        trvn_r = t_r * rvn_r;
        tp_l = t_l * p_l;
        tp_r = t_r * p_r;

        % the HLL flux
        Fn(1) = trvn_l * rho_l + trvn_r * rho_r;
        Fn(2:dim+1) = trvn_l * m_l(:) + trvn_r * m_r(:) + (tp_l + tp_r) * normal(:);
        Fn(dim+2) = trvn_l * E_l + trvn_r * E_r + tp_l * vn_l + tp_r * vn_r;
        
        return;
    end
end

end
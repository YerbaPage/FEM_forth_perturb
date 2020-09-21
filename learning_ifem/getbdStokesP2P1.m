function [AD,BD,f,g,u,p,ufreeDof,pDof] = getbdStokesP2P1
%% Boundary condition of Stokes equation: P2-P0 elements

%% Initial set up
%     f = [f1; f2];    % set in Neumann boundary condition
g = zeros(Np,1);
u = zeros(2*Nu,1);    
p = zeros(Np,1);
ufreeDof = (1:Nu)';
pDof = (1:Np)';

if ~exist('bdFlag','var'), bdFlag = []; end
if ~isfield(pde,'g_D'), pde.g_D = []; end
if ~isfield(pde,'g_N'), pde.g_N = []; end
if ~isfield(pde,'g_R'), pde.g_R = []; end

%% Part 1: Find Dirichlet dof and modify the matrix
% Find Dirichlet boundary dof: fixedDof and pDof
isFixedDof = false(Nu,1);     
if ~isempty(bdFlag)       % case: bdFlag is not empty 
    elem2edge = elem2dof(:,4:6)-N;
    isDirichlet(elem2edge(bdFlag(:)==1)) = true;
    isFixedDof(edge(isDirichlet,:)) = true;   % nodes of all D-edges
    isFixedDof(N + find(isDirichlet')) = true;% dof on D-edges
    fixedDof = find(isFixedDof);
    ufreeDof = find(~isFixedDof);            
end
if isempty(bdFlag) && ~isempty(pde.g_D) && isempty(pde.g_N) && isempty(pde.g_R)
    fixedDof = bdDof; 
    isFixedDof(fixedDof) = true;
    ufreeDof = find(~isFixedDof);    
end
if isempty(fixedDof) % pure Neumann boundary condition
    % pde.g_N could be empty which is homogenous Neumann boundary condition
    fixedDof = 1;
    ufreeDof = (2:Nu)';    % eliminate the kernel by enforcing u(1) = 0;
end

% Modify the matrix
% Build Dirichlet boundary condition into the matrix AD by enforcing
% AD(fixedDof,fixedDof)=I, AD(fixedDof,ufreeDof)=0, AD(ufreeDof,fixedDof)=0.
% BD(:,fixedDof) = 0 and thus BD'(fixedDof,:) = 0.
bdidx = zeros(2*Nu,1); 
bdidx([fixedDof; Nu+fixedDof]) = 1;
Tbd = spdiags(bdidx,0,2*Nu,2*Nu);
T = spdiags(1-bdidx,0,2*Nu,2*Nu);
AD = T*A*T + Tbd;
BD = B*T;

%% Part 2: Find boundary edges and modify the right hand side f and g
% Find boundary edges: Neumann and Robin
Neumann = []; Robin = []; %#ok<*NASGU>
if ~isempty(bdFlag)
    isNeumann(elem2edge((bdFlag(:)==2)|(bdFlag(:) == 3))) = true;
    isRobin(elem2edge(bdFlag(:)==3)) = true;
    Neumannidx = find(isNeumann);        
    Neumann   = edge(isNeumann,:);
    Robin     = edge(isRobin,:);
end
if isempty(bdFlag) && (~isempty(pde.g_N) || ~isempty(pde.g_R))
    % no bdFlag, only pde.g_N or pde.g_R is given in the input
    Neumann = edge(bdDof>N,:);
    if ~isempty(pde.g_R)
        Robin = Neumann;
    end
end

% Neumann boundary condition
if ~isempty(pde.g_N) && ~isempty(Neumann) && ~(isnumeric(pde.g_N) && (pde.g_N == 0))
    [lambda,w] = quadpts1(3);
    nQuad = size(lambda,1);
    % quadratic bases (1---3---2)
    bdphi(:,1) = (2*lambda(:,1)-1).*lambda(:,1);
    bdphi(:,2) = (2*lambda(:,2)-1).*lambda(:,2);
    bdphi(:,3) = 4*lambda(:,1).*lambda(:,2);
    % length of edge
    ve = node(Neumann(:,1),:) - node(Neumann(:,2),:);
    edgeLength = sqrt(sum(ve.^2,2));
    % update RHS
    gex = zeros(size(Neumann,1),2);   % x-component
    gey = zeros(size(Neumann,1),2);   % y-component
    for pp = 1:nQuad
        pxy = lambda(pp,1)*node(Neumann(:,1),:)+lambda(pp,2)*node(Neumann(:,2),:);
        gp = pde.g_N(pxy);
        gex(:,1) = gex(:,1) + w(pp)*edgeLength.*gp(:,1)*bdphi(pp,1);
        gex(:,2) = gex(:,2) + w(pp)*edgeLength.*gp(:,1)*bdphi(pp,2);
        gey(:,1) = gey(:,1) + w(pp)*edgeLength.*gp(:,2)*bdphi(pp,1);
        gey(:,2) = gey(:,2) + w(pp)*edgeLength.*gp(:,2)*bdphi(pp,2);
        f1(N+Neumannidx) = f1(N+Neumannidx) + w(pp)*edgeLength.*gp(:,1)*bdphi(pp,3); % interior bubble
        f2(N+Neumannidx) = f2(N+Neumannidx) + w(pp)*edgeLength.*gp(:,2)*bdphi(pp,3); % interior bubble
    end
    f1(1:N) = f1(1:N) + accumarray(Neumann(:), gex(:),[N,1]);
    f2(1:N) = f2(1:N) + accumarray(Neumann(:), gey(:),[N,1]);
end
f = [f1; f2];
% The case non-empty Neumann but g_N=[] corresponds to the zero flux
% boundary condition on Neumann edges and no modification is needed.

% Dirichlet boundary conditions
if ~isempty(fixedDof) && ~isempty(pde.g_D) && ~(isnumeric(pde.g_D) && (pde.g_D == 0))
    u1 = zeros(Nu,1);
    u2 = zeros(Nu,1);
    idx = (fixedDof > N);              % index of edge dof
    uD = pde.g_D(node(fixedDof(~idx),:));  % bd value at vertex dofs    
    u1(fixedDof(~idx)) = uD(:,1);
    u2(fixedDof(~idx)) = uD(:,2);
    bdEdgeIdx = fixedDof(idx)-N;
    bdEdgeMid = (node(edge(bdEdgeIdx,1),:)+node(edge(bdEdgeIdx,2),:))/2;
    uD = pde.g_D(bdEdgeMid);         % bd values at middle points of edges
    u1(fixedDof(idx)) = uD(:,1);
    u2(fixedDof(idx)) = uD(:,2);
    u = [u1; u2]; % Dirichlet bd condition is built into u
    f = f - A*u;  % bring affect of nonhomgenous Dirichlet bd condition to
    g = g - B*u;  % the right hand side
    g = g - mean(g);         
    f(fixedDof) = u1(fixedDof);
    f(fixedDof+Nu) = u2(fixedDof);
end
% The case non-empty Dirichlet but g_D=[] corresponds to the zero Dirichlet
% boundary condition and no modification is needed.

% modfiy pressure dof for pure Dirichlet
if isempty(Neumann)
    pDof = (1:Np-1)';
end

ufreeDof = [ufreeDof; Nu+ufreeDof];    
end % end of function getbdStokesP2P1
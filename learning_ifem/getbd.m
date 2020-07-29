function [AD,b,u,freeNode,isPureNeumann] = getbd(b)
%% Set up of boundary conditions.
%
% 1) Modify the matrix for Dirichlet boundary nodes, which are not degree
% of freedom. Values at these nodes are evaluatation of pde.g_D. The
% original stiffness matrix A is turn into the matrix AD by enforcing
% AD(fixedNode,fixedNode)=I, AD(fixedNode,freeNode)=0, AD(freeNode,fixedNode)=0.
%
% 2) Modify the right hand side b. The Neumann boundary integral is added
% to b. For Dirichlet boundary ndoes, b(fixedNode) is the evaluation of
% pde.g_D.
%
% Special attentation should be given for the pure Neumann boundary
% condition. To enforce the compatible condition, the vector b should have
% mean value zero. To avoid a singular matrix, the 1st node is chosen as
% fixedNode. 
%
% The order of assigning Neumann and Dirichlet boundary condition is
% important to get the right setting at the intersection nodes of Dirichlet
% and Neumann boundary edges.
%
% Reference: Long Chen. Finite Element Methods and its Programming. Lecture
% Notes.

u = zeros(Ndof,1); 
%% Initial check
if ~isfield(pde,'g_D'), pde.g_D = []; end
if ~isfield(pde,'g_N'), pde.g_N = []; end
if ~isfield(pde,'g_R'), pde.g_R = []; end

%% Part 1: Modify the matrix for Dirichlet and Robin condition
% Robin boundary condition
Robin = [];
isRobin = (bdFlag(:) == 3);
if any(isRobin)
    allEdge = [elem(:,[2,3]); elem(:,[3,1]); elem(:,[1,2])];
    Robin = allEdge(isRobin,:);
end
if ~isempty(Robin) && ~isempty(pde.g_R) && ~(isnumeric(pde.g_R) && (pde.g_R == 0))
    ve = node(Robin(:,1),:) - node(Robin(:,2),:);
    edgeLength = sqrt(sum(ve.^2,2)); 
    mid = (node(Robin(:,1),:) + node(Robin(:,2),:))/2;
    % use Simplson rule to compute int g_R phi_iphi_j ds
    ii = [Robin(:,1),Robin(:,1),Robin(:,2),Robin(:,2)];
    jj = [Robin(:,1),Robin(:,2),Robin(:,1),Robin(:,2)];
    temp = pde.g_R(mid).*edgeLength;
    ss = [1/3*temp, 1/6*temp, 1/6*temp, 1/3*temp];
    A = A + sparse(ii,jj,ss,Ndof,Ndof);
end

% Find Dirichlet boundary nodes: fixedNode
fixedNode = []; freeNode = [];
if ~isempty(bdFlag) % find boundary edges and boundary nodes
    [fixedNode,bdEdge,isBdNode] = findboundary(elem,bdFlag);
    freeNode = ~isBdNode;
end
if isempty(bdFlag) && ~isempty(pde.g_D) && isempty(pde.g_N) && isempty(pde.g_R)
    % no bdFlag, only pde.g_D is given
    [fixedNode,bdEdge,isBdNode] = findboundary(elem);
    freeNode = ~isBdNode;
end

% Modify the matrix for different boundary conditions 
% Dirichlet boundary condition
% Build Dirichlet boundary condition into the matrix AD by enforcing
% AD(fixedNode,fixedNode)=I, AD(fixedNode,freeNode)=0, AD(freeNode,fixedNode)=0.
if ~isempty(fixedNode)
    bdidx = zeros(Ndof,1); 
    bdidx(fixedNode) = 1;
    Tbd = spdiags(bdidx,0,Ndof,Ndof);
    T = spdiags(1-bdidx,0,Ndof,Ndof);
    AD = T*A*T + Tbd;
end
% Neumann boundary condition
isPureNeumann = false;
if isempty(fixedNode) && isempty(Robin) % pure Neumann boundary condition
    isPureNeumann = true;
    AD = A;
    AD(1,1) = AD(1,1) + 1e-6;
%         fixedNode = 1;
%         freeNode = 2:Ndof;    % eliminate the kernel by enforcing u(1) = 0;
end
% Robin boundary condition
if isempty(fixedNode) && ~isempty(Robin)
    AD = A;
end

%% Part 2: Find boundary edges and modify the right hand side b
% Find boundary edges: Neumann
Neumann = []; 
if ~isempty(bdFlag)  % bdFlag specifies different bd conditions
    Neumann = bdEdge;        
end
if isempty(bdFlag) && (~isempty(pde.g_N) || ~isempty(pde.g_R))
    % no bdFlag, only pde.g_N or pde.g_R is given in the input
    [tempvar,Neumann] = findboundary(elem); %#ok<ASGLU>
end

% Neumann boundary condition
if  isnumeric(pde.g_N) && all(pde.g_N == 0)
    pde.g_N = [];
end
if ~isempty(Neumann) && ~isempty(pde.g_N)
    el = sqrt(sum((node(Neumann(:,1),:) - node(Neumann(:,2),:)).^2,2));
    if ~isfield(option,'gNquadorder')
        option.gNquadorder = 2;   % default order exact for linear gN
    end
    [lambdagN,weightgN] = quadpts1(option.gNquadorder);
    phigN = lambdagN;                 % linear bases
    nQuadgN = size(lambdagN,1);
    ge = zeros(size(Neumann,1),2);
    for pp = 1:nQuadgN
        % quadrature points in the x-y coordinate
        ppxy = lambdagN(pp,1)*node(Neumann(:,1),:) ...
             + lambdagN(pp,2)*node(Neumann(:,2),:);
        gNp = pde.g_N(ppxy);
        for igN = 1:2
            ge(:,igN) = ge(:,igN) + weightgN(pp)*phigN(pp,igN)*gNp;
        end
    end
    ge = ge.*repmat(el,1,2);
    b = b + accumarray(Neumann(:), ge(:),[Ndof,1]); 
end
% The case with non-empty Neumann edges but g_N=0 or g_N=[] corresponds to
% the zero flux boundary condition on Neumann edges and no modification of
% A,u,b is needed.

% Dirichlet boundary condition
if isnumeric(pde.g_D) && all(pde.g_D == 0)   % zero g_D
    pde.g_D = [];
end
if ~isPureNeumann && ~isempty(fixedNode) && ~isempty(pde.g_D)
    if isnumeric(pde.g_D)  % pde.g_D could be a numerical array 
        u(fixedNode) = pde.g_D(fixedNode); 
    else % pde.g_D is a function handle
        u(fixedNode) = pde.g_D(node(fixedNode,:));
    end
    b = b - A*u;
end
if ~isempty(fixedNode) % non-empty Dirichlet boundary condition
    b(fixedNode) = u(fixedNode);
end
% The case with non-empty Dirichlet nodes but g_D=0 or g_D=[] corresponds
% to the zero Dirichlet boundary condition and no modification of u,b is
% needed.

% Pure Neumann boundary condition
if isPureNeumann
    b = b - mean(b); % compatiable condition (f,1) + <gN,1> = 0
%         b(1) = 0;
end
end % end of getbd
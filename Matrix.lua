local random = math.random
local Matrix = {}

Matrix.__index = Matrix

function Matrix:new(n, m, value)
  local self = {}

  self.matrix = {}

  self.n = n or 4
  self.m = m or 4

  for i = 1, self.n do
    self.matrix[i] = {}
    for j = 1, self.m do
      self.matrix[i][j] = value or 0
    end
  end

  return setmetatable(self, Matrix)
end

function Matrix:get(i, j)
  return self.matrix[i][j]
end

function Matrix:set(i, j, value)
  self.matrix[i][j] = value
end

function Matrix:apply(func)
  local newMatrix = Matrix:new(self.n, self.m)

  for i = 1, self.n do
    for j = 1, self.m do
      newMatrix.matrix[i][j] = func(self.matrix[i][j])
    end
  end

  return newMatrix
end

function Matrix:transpose()
  local newMatrix = Matrix:new(self.m, self.n)

  for i = 1, self.m do
    for j = 1, self.n do
      newMatrix.matrix[i][j] = self.matrix[j][i]
    end
  end

  return newMatrix
end

function Matrix:dot(o)

  local newMatrix = Matrix:new(self.n, o.m)

  local sum;

  for i = 1, self.n do
    for j = 1, o.m do
      sum = 0
      for k = 1, self.m do
        sum = sum + self.matrix[i][k] * o.matrix[k][j]
        newMatrix.matrix[i][j] = sum
      end

    end
  end

  return newMatrix
end

function Matrix:add(o)
  local newMatrix = Matrix:new(self.n, self.m)

  for i = 1, self.n do
    for j = 1, self.m do
      newMatrix.matrix[i][j] = self.matrix[i][j] + o.matrix[i][j]
    end
  end

  return newMatrix
end

function Matrix:subtract(o)
  local newMatrix = Matrix:new(self.n, self.m)

  for i = 1, self.n do
    for j = 1, self.m do
      newMatrix.matrix[i][j] = self.matrix[i][j] - o.matrix[i][j]
    end
  end

  return newMatrix
end

function Matrix:randomize()
  for i = 1, self.n do
    self.matrix[i] = {}
    for j = 1, self.m do
      self.matrix[i][j] = (math.random() - math.random()) * 1
    end
  end
end

function Matrix:multNum(x)
  local newMatrix = Matrix:new(self.n, self.m)
  for i = 1, self.n do
    for j = 1, self.m do
      newMatrix.matrix[i][j] = self.matrix[i][j] * x
    end
  end

  return newMatrix
end

function Matrix:addNum(x)
  local newMatrix = Matrix:new(self.n, self.m)
  for i = 1, self.n do
    for j = 1, self.m do
      newMatrix.matrix[i][j] = self.matrix[i][j] + x
    end
  end

  return newMatrix
end

function Matrix:eDot(o) -- Hadamard product
  local newMatrix = Matrix:new(self.n, self.m)
  for i = 1, self.n do
    for j = 1, self.m do
      newMatrix.matrix[i][j] = self.matrix[i][j] * o.matrix[i][j]
    end
  end

  return newMatrix
end

function Matrix:toString()
  local s = ""

  for i = 1, self.n do
    for j = 1, self.m do
      s = s .. self.matrix[i][j] .. "\t"
    end
    s = s .. "\n"
  end

  return s
end

function Matrix:__mul(o)
  if(type(o) == "number") then
    return self:multNum(o)
  end

  return self:dot(o)
end

function Matrix:__div(o)
  return self:multNum(1 / o)
end

function Matrix:__add(o)
  if(type(o) == "number") then
    return self:addNum(o);
  end
  return self:add(o)
end

function Matrix:__sub(o)
  return self:subtract(o)
end

function Matrix:__tostring()
  return self:toString()
end


return Matrix

import argparse
import hashlib
import math
import os
import platform
import secrets
import socket
import sys
import time
import uuid

# Gregorian epoch as nanoseconds before unix epoch
GREGORIAN_EPOCH_NS: int = -12219292800000000000
B36_VOCAB: str = "0123456789abcdefghijklmnopqrstuvwxyz"
B64_VOCAB: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"	# non-standard btw

def baseencode(n: int, base: int, vocab: str) -> str:
	"""
	Encode in an arbitrary base from base 10
	"""
	n = int(n)
	base = int(base)
	vocab = vocab[:base]

	if n == 0:
		return "0"

	if len(vocab) < 2:
		raise ValueError("Weird bases not supported")

	if len(vocab) < base:
		raise ValueError("Vocab must be at least of size n, where n is base")

	return "".join(vocab[n // base ** i % base] for i in range(int(math.log(n, base)) + 1))[::-1]

def basedecode(n: int | str, base: int, vocab: str) -> int:
	"""
	Decode from arbitrary base to base 10
	"""

	if isinstance(n, int):
		return n
	if len(vocab) < 2:
		raise ValueError("Weird bases not supported")
	if len(vocab) < base:
		raise ValueError("Vocab must be at least of size n, where n is base")

	vocab = vocab[:base]
	length: int = len(n)

	return sum(vocab.find(c) * base ** (length - i - 1) for i, c in enumerate(n))

class CustomID():
	def __init__(self, ident: int | str, base: int = 10, vocab: str = B36_VOCAB):
		if isinstance(ident, str):
			self.ident: int = basedecode(ident, base, vocab)
		else:
			self.ident: int = ident

	@property
	def hex(self) -> str:
		return hex(self.ident)

	@property
	def bytes(self):
		return self.ident.to_bytes()

	@property
	def bytes_le(self):
		return self.ident.to_bytes(byteorder="little")

	def __str__(self):
		return baseencode(self.ident, 64, B64_VOCAB)

def now() -> int:
	return time.time_ns() // 1_000_000

def cuid1(prefix: str = "c", count: int = 0, timestamp: int | None = None) -> str:
	"""
	Return CUID1
	"""

	def fingerprint() -> str:
		"""
		fingies!!
		"""

		hostname: str = platform.node()
		host_num: int = sum(ord(char) for char in hostname) + len(hostname) + 36

		host_b36: str = baseencode(host_num, 36, B36_VOCAB)[:2].rjust(2, "0")
		pid_part: str = baseencode(os.getpid(), 36, B36_VOCAB)[:2].rjust(2, "0")

		return pid_part + host_b36

	time_str: str = baseencode(timestamp or now(), 36, B36_VOCAB)[:8].rjust(8, "0")
	count_str: str = baseencode(count, 36, B36_VOCAB)[:8].rjust(4, "0")
	fingerprint: str = fingerprint()
	random_str: str = baseencode(int(secrets.randbits(64)), 36, B36_VOCAB)[:8].rjust(8, "0")

	return prefix + time_str + count_str + fingerprint + random_str

def cuid2(prefix: str = "c", count: int = 0, timestamp: int | None = None) -> str:
	"""
	Return CUID2
	"""

	if timestamp is None:
		timestamp = now()

	def fingerprint() -> int:
		try:
			host: str = socket.gethostname()
		except (OSError, UnicodeDecodeError):
			host: str = platform.node() or "unknown"

		pid: int = os.getpid()
		raw: bytes = hashlib.sha256(f"{host}|{pid}".encode()).digest()
		fingie: int = int.from_bytes(raw[:2], "big")

		return fingie

	seed: str = f"{timestamp}{secrets.randbits(64)}{count}{fingerprint()}"
	hash = hashlib.sha3_256(seed.encode("utf-8"))
	hash_int: int = int(hash.hexdigest(), 16)
	hash_b36: str = baseencode(hash_int, 36, B36_VOCAB)

	return f"{prefix[0]}{hash_b36[:23]}"

def main() -> None:
	parser = argparse.ArgumentParser(
		description="CUID and UUID CLI utility",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
	%(prog)s
	%(prog)s -v uuid4
	%(prog)s --version UUIDv6

Available UUID versions:
	UUIDv1, UUIDv3, UUIDv4, UUIDv5 UUIDv6, UUIDv7, UUIDv8
Available CUID versions:
	CUID1, CUID2
Other IDs:
	random (fully random string)
	rtime (8, 16, 32, or 64-bit milliseconds from Unix epoch; rest is random)
		"""
	)
	parser.add_argument(
		"-v", "--version", type=str.casefold, default="random", choices=[
			"random", "rtime", "cuid1", "cuid", "c1", "c", "cuid2", "c2",
			"uuid1", "uuid3", "uuid4", "uuid5", "uuid6", "uuid7", "uuid8",
			"uuidv1", "uuidv2", "uuidv3", "uuidv4", "uuidv5", "uuidv6", "uuidv7", "uuidv8"
		],
		help="Specify CUID or UUID version (default CUID; not case sensitive)"
	)
	parser.add_argument(
		"-c", "--count", type=int, default=0,
		help="Counter for count-based IDs"
	)
	parser.add_argument(
		"-b", "--bits", type=int,
		help="Minimum bits of entropy for random strings(overrides length, ignored otherwise)"
	)
	parser.add_argument(
		"-l", "--length", type=int, default=24,
		help="Length for random strings (default 24; ignored otherwise)"
	)
	parser.add_argument(
		"--name", type=str,
		help="Name for name-scoped IDs (required for UUIDv3 and v5; ignored otherwise)"
	)
	parser.add_argument(
		"--namespace", type=str.casefold,
		help="Namespace for namespace-scoped IDs (required UUIDv3 and v5; ignored otherwise)"
	)
	parser.add_argument(
		"-n", "--node", type=int,
		help="Node as 48-bit MAC address for hardware-identified IDs (ignored otherwise)"
	)
	parser.add_argument(
		"-t", "--timestamp", type=int,
		help="Timestamp in Unix millis for time-ordered IDs (ignored otherwise; overrides time-ns)"
	)
	parser.add_argument(
		"--ns", "--time-ns", type=int,
		help="Timestamp in nanoseconds since Unix epoch for time-ordered IDs (ignored otherwise)"
	)
	parser.add_argument(
		"--vocab", "--vocabulary", type=str, default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_",
		help="Allowed chars in random string (default 0-9 + alphanumeric; ignored otherwise)"
	)
	parser.add_argument(
		"-f", "--format", type=str.casefold, default="str", choices=[
			"str", "string", "base64", "b64", "base16", "b16", "hex", "bytes", "bytes-le", "b2",
			"b2le", "b"
		],
		help="Format to output ID as"
	)
	args = parser.parse_args()

	end: str = "\n" if sys.stdout.isatty() else ""

	# make sure timestamp has higher prio
	time_ns: int | None = args.timestamp * 1_000_000 if args.timestamp else args.ns
	time_ms: int | None= args.timestamp or args.ns // 10_000 if args.ns else None

	length: int = int(math.log(2 ** args.bits, len(args.vocab))) if args.bits else args.length
	bits: int = args.bits or int(math.ceil(math.log2(len(args.vocab) ** args.length)))

	match args.version:
		case "uuid1" | "uuidv1":
			node: int = args.node or uuid.getnode()
			gregorian_ns: int | None = time_ns - GREGORIAN_EPOCH_NS if time_ns else None
			output = uuid.uuid1(node=node, clock_seq=gregorian_ns)
		case "uuid3" | "uuidv3":
			if args.namespace is None:
				raise ValueError("namespace is required for UUIDv3")
			namespace = uuid.UUID(bytes=hashlib.md5(args.namespace.encode("utf-8")).digest())
			if args.name is None:
				raise ValueError("name is required for UUIDv3")
			output = uuid.uuid3(namespace, args.name)
		case "uuid4" | "uuidv4":
			output = uuid.uuid4()
		case "uuid5" | "uuidv5":
			if args.namespace is None:
				raise ValueError("namespace is required for UUIDv5")
			namespace = uuid.UUID(bytes=hashlib.md5(args.namespace.encode("utf-8")).digest())
			if args.name is None:
				raise ValueError("name is required for UUIDv3")
			output = uuid.uuid5(namespace, args.name)
		case "uuid6" | "uuidv6":
			timestamp: int | None = (time_ns - GREGORIAN_EPOCH_NS) // 100 if time_ns else None
			output = uuid.uuid6(args.node, timestamp)
		case "uuid7" | "uuidv7":
			output = uuid.uuid7()
		case "uuid8" | "uuidv8":
			output = uuid.uuid8()
		case "c" | "c1" | "cuid" | "cuid1":
			output = CustomID(cuid1("c", args.count, time_ms))
		case "c2" | "cuid2":
			output = CustomID(cuid2("c", args.count, time_ms))
		case "rtime":
			if bits < 16:
				raise ValueError("Too little entropy")
			tb: int = max([max(b, bits // 2) for b in [8, 16, 32, 64]])
			time_bits: int = (time_ms or now()) & ((1 << tb) - 1)
			time_bits = time_bits << bits - tb
			entropy_bits: int = secrets.randbits(bits - tb)
			result: int = time_bits & entropy_bits
			output = CustomID(baseencode(result, len(args.vocab), args.vocab))
		case "random":
			if bits < 16:
				raise ValueError("Too little entropy")
			output = CustomID("".join(secrets.choice(args.vocab) for _ in range(length)))
		case _:
			raise ValueError("Invalid UUID/CUID version")
	match args.format:
		case "base16" | "b16" | "hex":
			print(output.hex, end=end)
			return
		case "base64" | "b64":
			pass
		case "bytes" | "b2" | "b":
			print(output.bytes, end=end)
			return
		case "bytes_le" | "b2le":
			print(output.bytes_le, end=end)
			return
		case _:
			print(str(output), end=end)
			return

def cli_entry_point() -> None:
	main()

if __name__ == "__main__":
	main()

